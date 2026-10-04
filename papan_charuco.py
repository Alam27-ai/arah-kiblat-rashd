"""
papan_charuco.py — Modul bersama untuk papan kalibrasi ChArUco BayangKiblat.

Satu sumber kebenaran untuk parameter papan (ukuran, susunan, kamus ArUco)
dan seluruh logika deteksi/pengukuran yang bergantung padanya. Dipakai oleh:

  - cek_mutu_foto.py   (alat baris perintah, dua mode: kalibrasi & ukur)
  - kalibrasi_kamera.py (skrip kalibrasi kamera berdiri sendiri)
  - app.py             (menu "Ukur dari Foto" — pengukuran sudut interaktif)

Kenapa dipisah jadi modul sendiri: sebelumnya SQUARE_MM/NX/NY didefinisikan
dobel di kalibrasi_kamera.py dan cek_mutu_foto.py — risiko dua berkas itu
diam-diam tidak sinkron kalau ukuran papan berubah. Sekarang cukup ubah di
satu tempat.

Dua kelompok fungsi:
  1. Deteksi & kontrol mutu (sudah ada sebelumnya di cek_mutu_foto.py,
     dipindah ke sini tanpa mengubah perilaku).
  2. Rektifikasi ortho + garis Kiblat virtual (BARU — dipakai menu
     "Ukur dari Foto" di app.py).
"""

from __future__ import annotations

import itertools
import math
import os

import cv2
import numpy as np

# --- HARUS SESUAI PAPAN YANG DICETAK -----------------------------------------
VERSI_MODUL = "2026-10-04-template-geo"   # dicek app.py; ganti tiap ubah modul
SQUARE_MM = 30.0
MARKER_MM = 22.0
NX, NY = 6, 7
ARUCO_DICT = cv2.aruco.DICT_4X4_50
# -----------------------------------------------------------------------------

MIN_CORNERS = 8                    # ambang longgar, dipakai kalibrasi_kamera.py
MIN_CORNERS_UKUR = 12              # ambang lebih ketat untuk mode ukur/aplikasi
TOTAL_CORNERS = (NX - 1) * (NY - 1)
GRID_W, GRID_H = NX - 1, NY - 1

BOARD_W_MM = NX * SQUARE_MM        # bentang FISIK papan penuh (bukan cuma sudut interior)
BOARD_H_MM = NY * SQUARE_MM

# Titik pangkal gnomon TETAP (papan v2) — satu kotak masuk dari sisi kiri-bawah.
# Dipilih dekat satu tepi (bukan tengah/pojok) supaya bayangan pada altitude
# 15-60 deg umumnya masih tersapu di dalam grid (lihat RENCANA_CV.md §naskah
# kalibrasi_kamera). Karena posisinya tetap & tercetak, aplikasi tidak perlu
# lagi meminta pengguna mengklik titik ini di setiap foto.
GNOMON_TETAP_MM = (SQUARE_MM, BOARD_H_MM - SQUARE_MM)   # (30.0, 180.0)

AMBANG_AMAN = 0.10                 # derajat — di bawah ini foto dianggap baik
AMBANG_ULANG = 0.25                # derajat — di atas ini minta foto ulang

EXTS = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG",
        "*.heic", "*.HEIC", "*.dng", "*.DNG")


# =============================================================================
# 1. Deteksi & kontrol mutu (dipindah dari cek_mutu_foto.py, perilaku sama)
# =============================================================================

def build_detector():
    adict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard((NX, NY), SQUARE_MM, MARKER_MM, adict)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    det = cv2.aruco.CharucoDetector(board, cv2.aruco.CharucoParameters(), params)
    return adict, board, params, det


def scale_mm_per_px(corners, ids) -> float:
    """Skala citra dari jarak median antar sudut bertetangga dalam satu baris."""
    if corners is None or len(corners) < 6:
        return float("nan")
    pts = corners.reshape(-1, 2)
    idx = ids.flatten()
    d = []
    for a in range(len(idx)):
        for b in range(len(idx)):
            if idx[b] - idx[a] == 1 and idx[a] // GRID_W == idx[b] // GRID_W:
                d.append(np.linalg.norm(pts[b] - pts[a]))
    return SQUARE_MM / float(np.median(d)) if d else float("nan")


def illumination_stats(gray, corners):
    """Statistik iluminasi HANYA di dalam poligon papan, bukan seluruh frame."""
    if corners is None or len(corners) < 6:
        return (float("nan"),) * 3
    hull = cv2.convexHull(corners.reshape(-1, 1, 2).astype(np.float32))
    mask = np.zeros(gray.shape, np.uint8)
    cv2.fillConvexPoly(mask, hull.astype(np.int32), 255)
    v = gray[mask > 0]
    clip = 100.0 * float(np.mean(v >= 250))
    dark = 100.0 * float(np.mean(v <= 40))
    contrast = float(np.percentile(v, 95) - np.percentile(v, 5))
    return clip, dark, contrast


def cek_foto_ukur(path_or_array, verbose: bool = False) -> dict:
    """Periksa satu foto pengukuran TANPA perlu data kalibrasi kamera.

    Prinsip (lihat catatan panjang di cek_mutu_foto.py untuk latar belakang):
    sisa fit homografi papan->citra dikonversi langsung ke bias sudut dalam
    derajat, memakai foto itu sendiri saja. Menangkap distorsi lensa yang tak
    terserap, kertas melengkung, papan tidak rata, dan derau deteksi sekaligus.

    `path_or_array` boleh berupa path berkas ATAU array BGR (dari
    st.file_uploader / cv2.imdecode) — supaya bisa dipanggil langsung dari
    app.py tanpa menulis ke disk dulu.
    """
    adict, board, params, det = build_detector()
    obj_all = board.getChessboardCorners()[:, :2]

    if isinstance(path_or_array, np.ndarray):
        img_bgr = path_or_array
        nama = "(unggahan)"
    else:
        img_bgr = cv2.imread(path_or_array)
        nama = os.path.basename(path_or_array)
    if img_bgr is None:
        return {"file": nama, "vonis": "ULANGI", "alasan": "berkas tak terbaca"}

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    Him, Wim = gray.shape

    cc, ci, _, _ = det.detectBoard(gray)
    n = 0 if ci is None else len(ci)
    out = {"file": nama, "n_sudut": n, "resolusi": f"{Wim}x{Him}",
           "_cc": cc, "_ci": ci, "_gray_shape": (Wim, Him)}

    if n < MIN_CORNERS_UKUR:
        out.update(vonis="ULANGI",
                   alasan=f"hanya {n} sudut terdeteksi (perlu >={MIN_CORNERS_UKUR})")
        if verbose:
            _cetak_vonis(out)
        return out

    img_pts = cc.reshape(-1, 2).astype(np.float64)
    obj_pts = obj_all[ci.flatten()].astype(np.float64)

    Hm, _ = cv2.findHomography(obj_pts, img_pts, 0)
    proj = cv2.perspectiveTransform(obj_pts.reshape(1, -1, 2), Hm).reshape(-1, 2)
    res = img_pts - proj
    sisa_px = float(np.sqrt((res ** 2).sum(1).mean()))

    v = img_pts - np.array([Wim / 2, Him / 2])
    r = np.linalg.norm(v, axis=1)
    radial = (res * (v / r[:, None])).sum(1)
    rn = r / r.max()
    A = np.stack([rn ** 3, rn ** 5], 1)
    coef, *_ = np.linalg.lstsq(A, radial, rcond=None)
    fit = A @ coef
    denom = float(np.sum((radial - radial.mean()) ** 2))
    r2 = 1.0 - float(np.sum((radial - fit) ** 2)) / max(denom, 1e-12)
    radial_px = float(np.sqrt(np.mean(fit ** 2)))

    Hinv, _ = cv2.findHomography(img_pts, obj_pts, 0)
    back = cv2.perspectiveTransform(img_pts.reshape(1, -1, 2), Hinv).reshape(-1, 2)
    errs = []
    for a, b in itertools.combinations(range(len(obj_pts)), 2):
        L = np.linalg.norm(obj_pts[b] - obj_pts[a])
        if not (70.0 < L < 140.0):
            continue
        d0 = obj_pts[b] - obj_pts[a]
        d1 = back[b] - back[a]
        e = math.degrees(math.atan2(d1[1], d1[0]) - math.atan2(d0[1], d0[0]))
        errs.append((e + 180.0) % 360.0 - 180.0)
    bias = float(np.sqrt(np.mean(np.square(errs)))) if errs else float("nan")

    span = float(max(np.ptp(img_pts[:, 0]), np.ptp(img_pts[:, 1]))) / Wim
    ctr = img_pts.mean(axis=0)
    off = float(np.hypot(*(ctr - np.array([Wim / 2, Him / 2])))
                / np.hypot(Wim / 2, Him / 2))
    mmpx = scale_mm_per_px(cc, ci)
    clip, dark, contrast = illumination_stats(gray, cc)

    out.update(sisa_px=sisa_px, radial_px=radial_px, r2_radial=r2,
               bias_deg=bias, isi_frame=span, off_center=off,
               mm_per_px=mmpx, clip_pct=clip, gelap_pct=dark,
               kontras=contrast)

    alasan = []
    if bias > AMBANG_ULANG:
        alasan.append(f"bias sudut {bias:.3f}deg melebihi ambang {AMBANG_ULANG}deg")
    if span < 0.35:
        alasan.append(f"papan terlalu kecil di frame ({span*100:.0f}% lebar, "
                      "target >=50%) — dekatkan kamera")
    if off > 0.45:
        alasan.append(f"papan jauh dari pusat frame ({off:.2f}) — "
                      "distorsi lensa terbesar di tepi")
    # Catatan: piksel <=40 di papan sebagian besar adalah kotak/marker hitam
    # dan bayangan gnomon itu sendiri, jadi 'gelap_pct' bukan alasan ULANGI.
    # Cahaya kurang yang benar-benar merusak deteksi akan menaikkan bias_deg;
    # yang dijadikan gerbang keras hanya kontras yang sangat rendah.
    catatan = []
    if not np.isnan(contrast) and contrast < 60:
        alasan.append(f"kontras papan sangat rendah ({contrast:.0f}/255) — "
                      "foto terlalu gelap/berkabut, titik sudut tak andal")
    elif not np.isnan(dark) and dark > 60:
        catatan.append(f"{dark:.0f}% area papan gelap — cahaya kurang, "
                       "hasil masih dipakai karena bias sudut lolos")
    if not np.isnan(clip) and clip > 2:
        alasan.append(f"{clip:.0f}% area papan jenuh silau")
    if not np.isnan(mmpx) and mmpx > 0.40:
        alasan.append(f"skala kasar {mmpx:.2f} mm/px — foto terkompresi "
                      "atau kamera terlalu jauh")

    if alasan:
        out["vonis"] = "ULANGI"
    elif bias > AMBANG_AMAN or span < 0.50 or catatan:
        out["vonis"] = "SEDANG"
        alasan.extend(catatan)
        if bias > AMBANG_AMAN or span < 0.50:
            alasan.append("layak dipakai, tapi bisa jauh lebih baik dengan "
                          "mendekatkan kamera")
    else:
        out["vonis"] = "AMAN"
    out["alasan"] = "; ".join(alasan) if alasan else "semua kriteria terpenuhi"

    if r2 > 0.5 and radial_px > 0.25:
        out["sumber"] = ("berpola radial -> distorsi lensa; letakkan papan "
                         "lebih ke tengah frame atau matikan lensa ultra-wide")
    elif sisa_px > 0.6:
        out["sumber"] = ("sisa besar tapi TIDAK radial -> kemungkinan kertas "
                         "melengkung / tidak menempel rata pada alas")
    else:
        out["sumber"] = "sisa di lantai derau — distorsi tidak terukur di foto ini"

    if verbose:
        _cetak_vonis(out)
    return out


def _cetak_vonis(o: dict) -> None:
    tanda = {"AMAN": "[OK]  ", "SEDANG": "[~]   ", "ULANGI": "[X]   "}
    print(f"{tanda.get(o['vonis'], '')}{o['file']}  ->  {o['vonis']}")
    if "bias_deg" in o:
        print(f"        bias sudut (100 mm) : {o['bias_deg']:.3f} deg")
        print(f"        sisa homografi      : {o['sisa_px']:.3f} px "
              f"(radial {o['radial_px']:.3f} px, r2={o['r2_radial']:.2f})")
        print(f"        papan mengisi       : {o['isi_frame']*100:.0f}% lebar "
              f"frame, off-center {o['off_center']:.2f}")
        print(f"        sumber sisa         : {o['sumber']}")
    print(f"        catatan             : {o['alasan']}")
    print()


# =============================================================================
# 2. Rektifikasi ortho + garis Kiblat virtual (BARU — untuk menu "Ukur dari Foto")
# =============================================================================
#
# Konvensi arah yang WAJIB konsisten dengan qibla_core.solve_instant():
#   arah == "kanan"  -> target Kiblat berada di sisi SEARAH JARUM JAM dari
#                       bayangan, dilihat dari atas (sama seperti sweep=+delta_a
#                       untuk session_key="pagi" di render_simulation()).
#   arah == "kiri"   -> berlawanan jarum jam.
#
# Rotasi dihitung di dalam bidang papan (koordinat mm ChArUco), BUKAN azimuth
# kompas — karena itulah keunggulan metode ΔA: tidak perlu tahu Utara sejati.
# Yang penting citra ortho dirender tanpa flip (lihat rectify_ortho), sehingga
# "searah jarum jam seperti terlihat di citra" sama dengan "searah jarum jam
# dilihat dari atas papan sungguhan" — berlaku selama foto BUKAN dari kamera
# depan (selfie) yang di-mirror software-nya.

def rectify_ortho(img_bgr, cc, ci, board, px_per_mm: float = 4.0,
                  margin_mm: float = 15.0):
    """Warp foto asli menjadi tampak-atas metrik (skala piksel tetap & diketahui).

    Kembalikan (ortho_bgr, H_asli_ke_ortho, meta) dengan
    meta = {"px_per_mm", "margin_mm", "W_mm", "H_mm"} — dipakai fungsi lain
    di modul ini untuk mengonversi klik piksel <-> mm papan.
    """
    obj_full = np.array([[0, 0], [BOARD_W_MM, 0],
                         [BOARD_W_MM, BOARD_H_MM], [0, BOARD_H_MM]], np.float64)
    obj_pts = board.getChessboardCorners()[:, :2][ci.flatten()].astype(np.float64)
    img_pts = cc.reshape(-1, 2).astype(np.float64)
    H_mm_ke_img, _ = cv2.findHomography(obj_pts, img_pts, 0)
    if H_mm_ke_img is None:
        raise ValueError("Homografi gagal dihitung — sudut ChArUco terlalu sedikit/segaris.")

    W_ortho = int((BOARD_W_MM + 2 * margin_mm) * px_per_mm)
    H_ortho = int((BOARD_H_MM + 2 * margin_mm) * px_per_mm)
    dst = (obj_full + margin_mm) * px_per_mm
    src = cv2.perspectiveTransform(obj_full.reshape(1, -1, 2), H_mm_ke_img).reshape(-1, 2)
    M = cv2.getPerspectiveTransform(src.astype(np.float32), dst.astype(np.float32))
    ortho = cv2.warpPerspective(img_bgr, M, (W_ortho, H_ortho))

    meta = {"px_per_mm": px_per_mm, "margin_mm": margin_mm,
            "W_mm": BOARD_W_MM, "H_mm": BOARD_H_MM,
            "H_mm_ke_img": H_mm_ke_img, "M_img_ke_ortho": M,
            "img_wh": (int(img_bgr.shape[1]), int(img_bgr.shape[0]))}
    return ortho, meta


def arah_citra_gnomon(meta, O_mm, tinggi_mm: float = 60.0):
    """Perkiraan GEOMETRIS arah citra gnomon tegak di bidang ortho.

    Gnomon tegak tidak berada di bidang papan, jadi di citra ortho ia tampak
    sebagai guratan dari O menjauhi titik nadir kamera. Arah itu bisa
    dihitung dari homografi papan->citra + panjang fokus perkiraan (pose
    kamera dari H, Zhang 2000): titik O+(0,0,h) diproyeksikan ke citra lalu
    dikembalikan ke bidang papan. Tidak butuh warna/kilap gnomon, sehingga
    gnomon hitam polos (pulpen, paku dicat gelap) pun bisa disingkirkan.

    Panjang fokus tidak diketahui -> dicoba 0.6/0.8/1.1 x sisi panjang foto.
    Kembalikan (theta_deg, setengah_kerucut_deg) dengan theta searah jarum jam
    dari arah 'atas' papan, atau None bila foto hampir tegak lurus dari atas
    (citra gnomon pendek/arah tak stabil -> tidak perlu disingkirkan)."""
    H = meta.get("H_mm_ke_img")
    wh = meta.get("img_wh")
    if H is None or wh is None:
        return None
    w, h = wh
    O = np.asarray(O_mm, dtype=float)
    hasil = []
    for f_rel in (0.6, 0.8, 1.1):
        f = f_rel * max(w, h)
        K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
        B = np.linalg.inv(K) @ H
        n1, n2 = np.linalg.norm(B[:, 0]), np.linalg.norm(B[:, 1])
        r1, r2, t = B[:, 0] / n1, B[:, 1] / n2, B[:, 2] * 2.0 / (n1 + n2)
        if t[2] < 0:
            r1, r2, t = -r1, -r2, -t
        r3 = np.cross(r1, r2)
        sisi = 1.0 if -(r3 @ t) > 0 else -1.0          # sisi papan tempat kamera
        P = K @ (r1 * O[0] + r2 * O[1] + r3 * sisi * tinggi_mm + t)
        q = np.linalg.inv(H) @ np.array([P[0] / P[2], P[1] / P[2], 1.0])
        d = q[:2] / q[2] - O
        hasil.append((math.degrees(math.atan2(d[0], -d[1])) % 360, float(np.hypot(*d))))
    sudut = np.radians([a for a, _ in hasil])
    rerata = math.degrees(math.atan2(np.sin(sudut).mean(), np.cos(sudut).mean())) % 360
    sebar = max(min(abs(a - rerata), 360 - abs(a - rerata)) for a, _ in hasil)
    if hasil[1][1] < 10.0 or sebar > 30.0:
        return None
    return rerata, 25.0 + sebar


def _beda_sudut(a, b):
    d = abs(a - b) % 360
    return min(d, 360 - d)


def px_ortho_ke_mm(pt_px, meta) -> np.ndarray:
    x, y = pt_px
    return np.array([x / meta["px_per_mm"] - meta["margin_mm"],
                     y / meta["px_per_mm"] - meta["margin_mm"]])


def mm_ke_px_ortho(pt_mm, meta) -> np.ndarray:
    x, y = pt_mm
    return np.array([(x + meta["margin_mm"]) * meta["px_per_mm"],
                     (y + meta["margin_mm"]) * meta["px_per_mm"]])


def putar_vektor(v_mm: np.ndarray, delta_a_deg: float, arah: str) -> np.ndarray:
    """Putar vektor `v_mm` sejauh delta_a_deg, ke arah "kanan" (searah jarum
    jam seperti terlihat di citra ortho) atau "kiri" (berlawanan)."""
    theta = math.radians(delta_a_deg if arah == "kanan" else -delta_a_deg)
    c, s = math.cos(theta), math.sin(theta)
    R = np.array([[c, -s], [s, c]])   # rotasi CW-terlihat untuk koordinat y-ke-bawah
    return R @ v_mm


def hitung_pengukuran(O_mm: np.ndarray, B_mm: np.ndarray,
                      delta_a_deg: float, arah: str,
                      tinggi_gnomon_m: float, altitude_efemeris_deg: float,
                      ambang_altitude_deg: float = 0.5) -> dict:
    """Dari dua titik yang diklik (pangkal & ujung bayangan, dalam mm papan),
    hitung vektor arah Kiblat dan validasi silang altitude (§7.3 RENCANA_CV.md).
    """
    v_bayangan = B_mm - O_mm
    panjang_mm = float(np.linalg.norm(v_bayangan))
    v_kiblat = putar_vektor(v_bayangan, delta_a_deg, arah)

    alt_terukur = math.degrees(math.atan2(tinggi_gnomon_m, panjang_mm / 1000.0)) \
        if panjang_mm > 0 else float("nan")
    selisih_alt = alt_terukur - altitude_efemeris_deg
    alt_ok = abs(selisih_alt) <= ambang_altitude_deg

    return {
        "panjang_bayangan_mm": panjang_mm,
        "v_bayangan_mm": v_bayangan,
        "v_kiblat_mm": v_kiblat,
        "altitude_terukur_deg": alt_terukur,
        "altitude_efemeris_deg": altitude_efemeris_deg,
        "selisih_altitude_deg": selisih_alt,
        "validasi_altitude_ok": bool(alt_ok),
    }


def gambar_overlay(ortho_bgr, meta, O_mm, B_mm, v_kiblat_mm,
                   delta_a_deg, arah, panjang_garis_mm: float | None = None):
    """Gambar garis bayangan (merah), garis Kiblat (hijau), busur mini +
    label ΔA pada citra ortho. Mengembalikan citra baru (tidak mengubah input)."""
    vis = ortho_bgr.copy()
    if panjang_garis_mm is None:
        panjang_garis_mm = float(np.linalg.norm(B_mm - O_mm)) * 1.4

    O_px = mm_ke_px_ortho(O_mm, meta).astype(int)
    B_px = mm_ke_px_ortho(B_mm, meta).astype(int)
    v_kiblat_hat = v_kiblat_mm / (np.linalg.norm(v_kiblat_mm) + 1e-9)
    K_mm = O_mm + v_kiblat_hat * panjang_garis_mm
    K_px = mm_ke_px_ortho(K_mm, meta).astype(int)

    cv2.line(vis, tuple(O_px), tuple(B_px), (0, 0, 220), 3, cv2.LINE_AA)
    cv2.line(vis, tuple(O_px), tuple(K_px), (0, 170, 0), 3, cv2.LINE_AA)
    cv2.circle(vis, tuple(O_px), 6, (30, 30, 30), -1, cv2.LINE_AA)

    # busur mini di antara dua garis, radius kecil relatif panjang bayangan
    r_mm = min(panjang_garis_mm * 0.35, 40.0)
    v_bay_hat = (B_mm - O_mm) / (np.linalg.norm(B_mm - O_mm) + 1e-9)
    steps = max(int(abs(delta_a_deg)), 2)
    pts = []
    for k in range(steps + 1):
        vv = putar_vektor(v_bay_hat, delta_a_deg * k / steps, arah)
        p_mm = O_mm + vv * r_mm
        pts.append(mm_ke_px_ortho(p_mm, meta).astype(int))
    for p0, p1 in zip(pts[:-1], pts[1:]):
        cv2.line(vis, tuple(p0), tuple(p1), (40, 40, 40), 2, cv2.LINE_AA)

    label_px = mm_ke_px_ortho(O_mm + putar_vektor(v_bay_hat, delta_a_deg / 2, arah)
                              * (r_mm * 1.4), meta).astype(int)
    Him, Wim = vis.shape[:2]

    def _teks_latar(p, teks, skala, warna, tebal=2):
        """Tulis teks dengan latar putih tipis di belakangnya (supaya kontras
        di atas garis/kotak apa pun), posisi selalu dijaga di dalam kanvas."""
        (tw, th), base = cv2.getTextSize(teks, cv2.FONT_HERSHEY_SIMPLEX, skala, tebal)
        x = int(np.clip(p[0], 2, max(Wim - tw - 2, 2)))
        y = int(np.clip(p[1], th + 2, max(Him - base - 2, th + 2)))
        cv2.rectangle(vis, (x - 2, y - th - 2), (x + tw + 2, y + base + 2),
                     (255, 255, 255), -1)
        cv2.putText(vis, teks, (x, y), cv2.FONT_HERSHEY_SIMPLEX, skala,
                   warna, tebal, cv2.LINE_AA)

    # label ΔA di dekat busur
    _teks_latar(label_px, f"dA={delta_a_deg:.2f} {arah}", 0.55, (20, 20, 20))
    # label ujung garis: didorong keluar sepanjang arah garis itu sendiri (bukan
    # offset piksel tetap) supaya tidak pernah menimpa garisnya sendiri
    dir_bay = (B_px - O_px).astype(float)
    dir_bay = dir_bay / (np.linalg.norm(dir_bay) + 1e-9)
    _teks_latar(B_px + dir_bay * 14 + np.array([6, 0]), "bayangan", 0.5, (0, 0, 220))
    dir_kib = (K_px - O_px).astype(float)
    dir_kib = dir_kib / (np.linalg.norm(dir_kib) + 1e-9)
    _teks_latar(K_px + dir_kib * 14 + np.array([6, 0]), "KIBLAT", 0.55, (0, 140, 0))
    return vis


def titik_potong_tepi(O_mm: np.ndarray, v_mm: np.ndarray) -> dict:
    """Titik di mana sinar dari O sepanjang arah v_mm memotong tepi FISIK papan
    (persegi 0..BOARD_W_MM x 0..BOARD_H_MM — tepat di situ skala mm/cm papan v2
    dicetak). Dipakai untuk "Petunjuk Pemasangan Fisik": pengguna membaca satu
    angka di tepi papan, bukan menghitung sendiri.

    Kembalikan dict dengan sisi ('atas'/'bawah'/'kiri'/'kanan'), posisi_mm
    (jarak dari pojok kiri-atas SEPANJANG sisi itu), dan titik_mm (koordinat
    penuh, untuk keperluan lain seperti crop-zoom foto).
    """
    ox, oy = O_mm
    vx, vy = v_mm
    kandidat = []  # (t, sisi, posisi_mm)

    if abs(vx) > 1e-12:
        t = (0.0 - ox) / vx
        y = oy + t * vy
        if t > 1e-9 and -1e-6 <= y <= BOARD_H_MM + 1e-6:
            kandidat.append((t, "kiri", float(np.clip(y, 0, BOARD_H_MM))))
        t = (BOARD_W_MM - ox) / vx
        y = oy + t * vy
        if t > 1e-9 and -1e-6 <= y <= BOARD_H_MM + 1e-6:
            kandidat.append((t, "kanan", float(np.clip(y, 0, BOARD_H_MM))))

    if abs(vy) > 1e-12:
        t = (0.0 - oy) / vy
        x = ox + t * vx
        if t > 1e-9 and -1e-6 <= x <= BOARD_W_MM + 1e-6:
            kandidat.append((t, "atas", float(np.clip(x, 0, BOARD_W_MM))))
        t = (BOARD_H_MM - oy) / vy
        x = ox + t * vx
        if t > 1e-9 and -1e-6 <= x <= BOARD_W_MM + 1e-6:
            kandidat.append((t, "bawah", float(np.clip(x, 0, BOARD_W_MM))))

    if not kandidat:
        return {"sisi": None, "posisi_mm": float("nan"),
               "titik_mm": np.array([np.nan, np.nan]),
               "posisi_cm_str": "-"}

    t, sisi, posisi_mm = min(kandidat, key=lambda k: k[0])
    titik_mm = np.array([ox + t * vx, oy + t * vy])
    cm = posisi_mm / 10.0
    return {
        "sisi": sisi,
        "posisi_mm": posisi_mm,
        "titik_mm": titik_mm,
        "posisi_cm_str": f"{cm:.2f}".replace(".", ","),
    }


def crop_zoom_titik(img_bgr, meta_atau_H, titik_px, radius_px: int = 90,
                    skala_output: int = 4):
    """Potong & perbesar area di sekitar satu titik piksel — dipakai untuk
    menampilkan zoom di sekitar titik potong tepi, supaya pengguna tinggal
    mencocokkan visual dengan papan fisik alih-alih menghitung interpolasi
    sendiri. `titik_px` dalam koordinat citra `img_bgr` yang diberikan
    (ortho ATAU foto asli, keduanya boleh)."""
    h, w = img_bgr.shape[:2]
    x, y = int(titik_px[0]), int(titik_px[1])
    x0, x1 = max(x - radius_px, 0), min(x + radius_px, w)
    y0, y1 = max(y - radius_px, 0), min(y + radius_px, h)
    if x1 <= x0 or y1 <= y0:
        return img_bgr.copy()
    crop = img_bgr[y0:y1, x0:x1].copy()
    big = cv2.resize(crop, None, fx=skala_output, fy=skala_output,
                     interpolation=cv2.INTER_CUBIC)
    # tandai posisi titik yang sebenarnya (silang) di citra yang sudah diperbesar
    cx = (x - x0) * skala_output
    cy = (y - y0) * skala_output
    cv2.drawMarker(big, (cx, cy), (0, 0, 220), cv2.MARKER_CROSS, 28, 2, cv2.LINE_AA)
    return big


def warp_titik_ortho_ke_asli(pts_px_ortho, meta):
    """Petakan titik-titik dari koordinat piksel ortho balik ke piksel foto ASLI
    (yang belum diluruskan), supaya garis Kiblat bisa ditumpangkan langsung di
    atas foto asli juga (bukan cuma versi ortho)."""
    Minv = np.linalg.inv(meta["M_img_ke_ortho"])
    pts = np.array(pts_px_ortho, dtype=np.float64).reshape(1, -1, 2)
    back = cv2.perspectiveTransform(pts, Minv).reshape(-1, 2)
    return back


# ---------------------------------------------------------------------------
# TAHAP 1 — deteksi otomatis ujung bayangan (menggantikan klik manual)
# ---------------------------------------------------------------------------
# Rentang warna gnomon yang DISARANKAN untuk dicat/ditempel pada batang gnomon
# supaya bisa dipisahkan otomatis dari bayangannya sendiri. Magenta/marun
# dipilih karena praktis tidak pernah muncul di papan (hitam/putih/abu-abu)
# maupun pada bayangan mana pun (bayangan selalu berupa versi GELAP/desaturasi
# dari warna aslinya, tidak pernah bertambah warna). Rentang HSV OpenCV:
# H 0-179, S 0-255, V 0-255.
#
# PENTING (ditemukan lewat uji coba, bukan cuma teori): warnanya harus TETAP
# GELAP (V rendah, mis. marun/magenta tua, BUKAN pink cerah) — deteksi ini
# jalan dua lapis, dan lapis pertama (anomali gelap vs latar lokal) berjalan
# SEBELUM piksel dicek warnanya. Kalau cat gnomon terlalu terang, segmen itu
# tidak lagi lebih gelap dari kertas di sekitarnya, jadi malah GAGAL masuk
# mask sama sekali (bukan cuma salah diklasifikasi) — mask jadi terputus di
# situ dan ujung bayangan yang sebenarnya (di seberang segmen terang itu)
# jadi tidak tersambung ke titik O lagi. Diuji langsung: magenta tua (abu2
# ~40) bekerja sempurna, magenta cerah (abu2 ~100) merusak hasil total.
GNOMON_HSV_LO = np.array([140, 60, 20])
GNOMON_HSV_HI = np.array([175, 255, 130])


# --- Deteksi bayangan berbasis TEMPLATE papan (2026-10-04) --------------------
# Masalah yang diperbaiki: gnomon tegak TIDAK berada di bidang papan, sehingga
# di citra ortho ia tampak sebagai guratan memanjang dari O (efek perspektif).
# Guratan itu punya piksel gelap, sehingga metode "objek gelap terjauh dari O"
# bisa mengikuti GNOMON, bukan bayangannya (kasus nyata: foto Bojong, gnomon
# baut logam polos -> garis merah mengikuti baut, salah ~125 deg).
#
# Dasar fisika metode ini: bayangan HANYA MENGGELAPKAN papan, merata, di kotak
# putih maupun hitam. Gnomon (logam/cat) menghasilkan campuran piksel gelap
# DAN terang (kilap). Maka citra dibandingkan dengan pola papan yang DIKETAHUI
# (render ChArUco), dihitung peta penggelapan D = 1 - teramati/harapan, lalu
# setiap arah dari O dinilai dari penggelapannya yang konsisten; arah yang
# mengandung pencerahan (D < AMBANG_TERANG) ditolak sebagai benda, bukan bayangan.
# Tidak memerlukan gnomon berwarna.

AMBANG_TERANG = -0.10        # D di bawah ini = lebih terang dari kertas -> benda
FRAKSI_TERANG_MAKS = 0.03    # arah dengan >3% piksel terang ditolak
MIN_MARGIN_SKOR = 0.05       # selisih skor arah terbaik vs arah lain (>20 deg)
MIN_PISAH_GNOMON_DEG = 20.0  # bayangan & citra gnomon terlalu berimpit -> ulangi
TOL_PANJANG_EFEMERIS = 0.15  # |L - L_prediksi| / L_prediksi


def peta_penggelapan(ortho_bgr, meta, board=None, sigma_mm: float = 10.0,
                     erosi_mm: float = 1.0, sat_maks: int = 90,
                     kembalikan_putih: bool = False):
    """Peta D = 1 - abu2_teramati / abu2_harapan (bayangan > 0, kilap < 0).

    Harapan dihitung terpisah untuk kotak putih dan hitam dari template papan,
    dengan rata-rata lokal ternormalisasi (sigma ~ 1/3 kotak) sehingga
    gradien pencahayaan ikut terkoreksi. NaN di tepi kotak (tepi cetak tidak
    pernah pas sempurna), di luar papan, dan pada piksel berwarna jenuh
    (overlay/teks berwarna)."""
    if board is None:
        _, board, _, _ = build_detector()
    ppm, m = meta["px_per_mm"], meta["margin_mm"]
    gray = cv2.cvtColor(ortho_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(ortho_bgr, cv2.COLOR_BGR2HSV)
    sat, val = hsv[..., 1], hsv[..., 2]
    Wb, Hb = int(round(meta["W_mm"] * ppm)), int(round(meta["H_mm"] * ppm))
    tpl = board.generateImage((Wb, Hb), marginSize=0, borderBits=1)
    putih = np.zeros(gray.shape, np.uint8)
    pada_papan = np.zeros(gray.shape, np.uint8)
    x0 = y0 = int(round(m * ppm))
    putih[y0:y0 + Hb, x0:x0 + Wb] = (tpl > 127)
    pada_papan[y0:y0 + Hb, x0:x0 + Wb] = 1
    hitam = pada_papan & (1 - putih)
    r = max(int(round(erosi_mm * ppm)), 1)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    # Saturasi hanya bermakna pada piksel cukup terang: piksel gelap (kotak
    # hitam yang terkena bayangan) sering punya S tinggi karena derau warna,
    # dan dulu ikut dibuang sehingga arah bayangan dianggap "kosong".
    sah = ((sat < sat_maks) | (val < 90)) & (gray > 0)

    # Piksel kotak hitam yang jauh lebih terang dari kotak hitam lain (kilap
    # gnomon logam yang citranya menutupi kotak hitam) jangan ikut menaikkan
    # "harapan" kotak hitam — kalau ikut, sisa kotak hitam di sekitarnya
    # tampak "tergelapkan" dan terbaca sebagai bayangan palsu.
    _ph = (cv2.erode(hitam, k) > 0) & sah
    _pp = (cv2.erode(putih, k) > 0) & sah
    if _ph.any() and _pp.any():
        med_h, med_p = float(np.median(gray[_ph])), float(np.median(gray[_pp]))
        kilap_hitam = gray > med_h + 0.5 * (med_p - med_h)
    else:
        kilap_hitam = np.zeros(gray.shape, bool)

    def harapan(kelas, buang=None):
        msk = (cv2.erode(kelas, k) > 0) & sah
        if buang is not None:
            msk = msk & ~buang
        s = sigma_mm * ppm
        num = cv2.GaussianBlur(np.where(msk, gray, 0).astype(np.float32), (0, 0), s)
        den = cv2.GaussianBlur(msk.astype(np.float32), (0, 0), s)
        return num / np.maximum(den, 1e-3), msk

    Ep, mp = harapan(putih)
    Eh, mh = harapan(hitam, buang=kilap_hitam)
    mh = mh | ((cv2.erode(hitam, k) > 0) & sah & kilap_hitam)   # tetap dinilai (D<0 = kilap)
    E = np.where(putih > 0, Ep, Eh)
    D = np.where(mp | mh, 1.0 - gray / np.maximum(E, 1.0), np.nan)
    if kembalikan_putih:
        return D, mp
    return D


def _deteksi_bayangan_template(ortho_bgr, meta, O_mm, L_pred_mm=None,
                               board=None, r_min_mm: float = 6.0,
                               celah_mm: float = 2.0) -> dict:
    ppm, m = meta["px_per_mm"], meta["margin_mm"]
    O = np.asarray(O_mm, dtype=float)
    D, D_putih = peta_penggelapan(ortho_bgr, meta, board, kembalikan_putih=True)
    # Uji kilap hanya di kotak PUTIH: di kotak hitam rasio teramati/harapan
    # sangat berderau (penyebut kecil, pantulan kertas), sehingga bayangan
    # yang melintasi kotak hitam bisa salah ditolak sebagai "benda terang".
    D_kilap = np.where(D_putih, D, np.nan)

    def ambil(xmm, ymm):
        xs = np.round((xmm + m) * ppm).astype(int)
        ys = np.round((ymm + m) * ppm).astype(int)
        ok = (xs >= 0) & (xs < D.shape[1]) & (ys >= 0) & (ys < D.shape[0])
        v = np.full(np.shape(xmm), np.nan)
        v[ok] = D[ys[ok], xs[ok]]
        return v

    def ambil_kilap(xmm, ymm):
        xs = np.round((xmm + m) * ppm).astype(int)
        ys = np.round((ymm + m) * ppm).astype(int)
        ok = (xs >= 0) & (xs < D.shape[1]) & (ys >= 0) & (ys < D.shape[0])
        v = np.full(np.shape(xmm), np.nan)
        v[ok] = D_kilap[ys[ok], xs[ok]]
        return v

    def satuan(th):   # th: derajat searah jarum jam dari arah 'atas' papan (-y)
        t = math.radians(th)
        return np.array([math.sin(t), -math.cos(t)])

    hasil = {"tip_mm": None, "tip_px": None, "yakin": False, "pakai_warna": False,
             "ujung_ambang_ketat": False, "n_piksel_gnomon": 0,
             "n_piksel_bayangan": 0, "metode": "template", "flags": [],
             "mask_debug": np.clip(np.nan_to_num(D) * 400 + 128, 0, 255).astype(np.uint8)}

    r_atas = 0.7 * L_pred_mm if (L_pred_mm and L_pred_mm > r_min_mm + 8) else 40.0
    r = np.arange(r_min_mm, r_atas, 0.25)

    # 1) pindai semua arah (kecuali kerucut citra gnomon hasil geometri)
    geo = arah_citra_gnomon(meta, O)
    skor, terang = [], []
    for th in np.arange(0.0, 360.0, 0.5):
        if geo is not None and _beda_sudut(th, geo[0]) < geo[1]:
            continue
        u = satuan(th)
        v = ambil(O[0] + r * u[0], O[1] + r * u[1])
        if np.mean(np.isnan(v)) > 0.6:   # tepi marker/kotak = NaN; cukup >=40% sampel sah
            continue
        vk = ambil_kilap(O[0] + r * u[0], O[1] + r * u[1])
        f_terang = float(np.mean(vk[~np.isnan(vk)] < AMBANG_TERANG)) if np.any(~np.isnan(vk)) else 0.0
        terang.append((f_terang, th))
        if f_terang > FRAKSI_TERANG_MAKS:
            continue
        skor.append((float(np.nanpercentile(v, 20)), th))
    if not skor:
        hasil["flags"].append("tidak ada arah yang menyerupai bayangan")
        return hasil
    skor.sort(reverse=True)
    terbaik, th = skor[0]
    # Pembanding margin: arah lain >20 deg dari terpilih, dan bukan pinggiran
    # kerucut gnomon (pangkal gnomon yang lebar ikut menggelapkan sekitarnya).
    lain = [s for s, a in skor if _beda_sudut(a, th) > 20
            and (geo is None or _beda_sudut(a, geo[0]) > geo[1] + 10)]
    margin = terbaik - (lain[0] if lain else 0.0)

    # 2) haluskan arah: titik berat penggelapan melintang bayangan
    offs = np.arange(-6.0, 6.01, 0.25)
    for _ in range(3):
        u = satuan(th)
        n = np.array([-u[1], u[0]])
        sudut, bobot = [], []
        for ri in np.arange(r_min_mm + 2, r[-1], 0.5):
            p = O + ri * u
            v = np.clip(np.nan_to_num(ambil(p[0] + offs * n[0], p[1] + offs * n[1])), 0, None)
            if v.sum() > 1e-3:
                sudut.append(math.degrees(math.atan2((v * offs).sum() / v.sum(), ri)))
                bobot.append(ri)
        if sudut:
            th = (th + float(np.average(sudut, weights=bobot))) % 360

    # 3) ujung = titik setengah-tingkat-gelap (tengah penumbra); NaN dilewati
    u = satuan(th)
    rr = np.arange(r_min_mm, 400.0, 0.25)
    v = ambil(O[0] + rr * u[0], O[1] + rr * u[1])
    tingkat = float(np.nanmedian(v[(rr > r_min_mm + 6) & (rr < r[-1])]))
    akhir, bawah = None, 0.0
    for ri, vi in zip(rr, v):
        if ri < r_min_mm + 6 or np.isnan(vi):
            continue
        if vi > 0.5 * tingkat:
            akhir, bawah = ri, 0.0
        else:
            bawah += 0.25
            if bawah > celah_mm:
                break
    if akhir is None or not np.isfinite(tingkat) or tingkat <= 0:
        hasil["flags"].append("ujung bayangan tidak ditemukan")
        return hasil

    B = O + akhir * u
    if geo is not None:
        arah_gnomon = geo[0]
    else:
        arah_gnomon = max(terang)[1] if terang else None
    pisah = None if arah_gnomon is None else min(abs(arah_gnomon - th), 360 - abs(arah_gnomon - th))

    if margin < MIN_MARGIN_SKOR:
        hasil["flags"].append("arah bayangan ambigu")
    peringatan = []
    # Arah yang dipilih sendiri sudah lolos uji kilap, jadi berdekatan dengan
    # citra gnomon hanya PERINGATAN (periksa visual), bukan penolakan.
    if pisah is not None and (geo is not None or max(terang)[0] > 0.10) \
            and pisah < MIN_PISAH_GNOMON_DEG + (geo[1] if geo is not None else 0):
        peringatan.append("bayangan berdekatan dengan citra gnomon — periksa titik, "
                          "atau potret lebih tegak dari atas")
    if L_pred_mm and abs(akhir - L_pred_mm) / L_pred_mm > TOL_PANJANG_EFEMERIS:
        peringatan.append(f"panjang bayangan {akhir:.1f} mm vs prediksi efemeris {L_pred_mm:.1f} mm")

    tip_px = mm_ke_px_ortho(B, meta)
    hasil.update({
        "tip_mm": B, "tip_px": (int(round(tip_px[0])), int(round(tip_px[1]))),
        "yakin": not hasil["flags"], "theta_bayangan_deg": th, "panjang_mm": akhir,
        "tingkat_gelap": tingkat, "margin_skor": margin,
        "arah_gnomon_deg": arah_gnomon, "pisah_gnomon_deg": pisah,
        "peringatan": peringatan,
    })
    return hasil


def deteksi_otomatis_bayangan(ortho_bgr, meta, O_mm, L_pred_mm=None, **kw) -> dict:
    """Cari otomatis ujung bayangan (titik B).

    Urutan: (1) metode TEMPLATE papan (tidak butuh gnomon berwarna, menolak
    citra gnomon lewat kilapnya); bila gagal/ragu, (2) metode anomali-gelap
    lama (`_deteksi_bayangan_anomali`, memakai pemisah warna gnomon bila ada).
    `L_pred_mm` = tinggi_gnomon / tan(altitude efemeris), opsional tetapi
    disarankan: mempersempit pencarian dan memberi peringatan bila panjang
    bayangan tidak cocok."""
    t = _deteksi_bayangan_template(ortho_bgr, meta, O_mm, L_pred_mm=L_pred_mm)
    if t["yakin"]:
        return t
    lama = _deteksi_bayangan_anomali(ortho_bgr, meta, O_mm, **kw)
    lama["metode"] = "anomali (cadangan)"
    lama["flags"] = t["flags"]
    lama["peringatan"] = []
    return lama




def _deteksi_bayangan_anomali(ortho_bgr, meta, O_mm, radius_cari_mm: float = 200.0,
                              ambang_anomali: int = 20, ambang_ujung: int = 45,
                              warna_lo=GNOMON_HSV_LO, warna_hi=GNOMON_HSV_HI) -> dict:
    """Cari otomatis ujung bayangan (titik B), menggantikan klik manual.

    Tiga lapis:
    1) ANOMALI GELAP (ambang LONGGAR `ambang_anomali`) — bandingkan citra
       dengan "latar lokal"-nya sendiri (median blur berkernel > 1 kotak
       papan, sehingga corak hitam-putih papan ikut terhaluskan/hilang).
       Piksel yang lebih gelap dari latar lokalnya adalah kandidat
       gnomon+bayangan (termasuk ekor penumbra yang pudar). Dipakai HANYA
       untuk menentukan KEBERADAAN objek yang menempel ke titik O — ambang
       longgar sengaja dipertahankan di sini supaya sambungan tipis di ekor
       bayangan tidak membuat komponennya terputus.
    2) PEMISAH WARNA GNOMON — di dalam komponen itu, piksel yang cocok
       rentang warna gnomon (lihat GNOMON_HSV_LO/HI) disingkirkan dari
       pencarian ujung (itu batang gnomon, bukan bayangan).
    3) AMBANG UJUNG KETAT (`ambang_ujung`, BARU) — dari sisa piksel non-
       gnomon, titik ujung HANYA dicari di antara piksel yang cukup gelap
       (anomali > ambang_ujung, lebih ketat dari lapis 1). Ini mengatasi
       masalah nyata yang ditemukan lewat uji coba: ambang longgar tunggal
       ikut menghitung ekor penumbra (bayangan pudar/kabur di ujung, bukan
       inti bayangan tegas) sebagai bagian bayangan, sehingga panjang
       bayangan yang terbaca membengkak dan berubah-ubah tergantung
       ketajaman/kompresi foto sumber (foto beresolusi tinggi menampakkan
       ekor pudar itu lebih jelas daripada foto terkompresi, padahal objek
       fisiknya identik) — lihat catatan pengujian di percakapan/README.
       Kalau tidak ada piksel yang lolos ambang ketat ini (bayangan memang
       sangat pudar seluruhnya), turun ke ambang longgar sebagai fallback
       dan `ujung_ambang_ketat` diisi False supaya pemanggil tahu ini
       kurang meyakinkan.

    Kalau gnomon TIDAK diberi warna (foto lama, sebelum papan/gnomon dicat),
    lapis (2) tidak menemukan piksel berwarna gnomon sama sekali, sehingga
    seluruh komponen (minus ambang ketat di lapis 3) dipakai apa adanya —
    hasilnya kembali ke "ujung objek gelap tegas terjauh dari O" (gnomon+
    bayangan tercampur). Fungsi tetap jalan tapi kualitasnya lebih rendah —
    lihat `pakai_warna` untuk tahu mana yang terjadi.

    Kembalikan dict: tip_mm, tip_px, yakin (bool), pakai_warna (bool),
    ujung_ambang_ketat (bool), n_piksel_gnomon, n_piksel_bayangan,
    mask_debug (untuk ditampilkan ke pengguna sebagai QA visual).
    """
    # Redam dulu tekstur halus kertas/derau sensor SEBELUM dibandingkan ke
    # latar lokal (BARU — ditemukan lewat uji coba nyata: foto beresolusi
    # tinggi/asli kamera punya derau piksel-ke-piksel yang jauh lebih
    # terlihat daripada foto yang sudah terkompresi WhatsApp; tanpa peredaman
    # ini, derau itu ikut lolos ambang anomali di banyak tempat tersebar di
    # seluruh papan — bukan cuma di gnomon/bayangan — dan kalau kebetulan
    # tersambung (lewat closing di bawah) ke komponen gnomon/bayangan,
    # pencarian "titik terjauh dari O" bisa melompat jauh ke noise yang
    # sama sekali bukan bayangan. Blur ini membuat hasil pada foto resolusi
    # tinggi vs terkompresi jadi konsisten (diuji: ~69mm di kedua versi,
    # sebelumnya beda jauh 69mm vs 85mm).
    gray0 = cv2.cvtColor(ortho_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray0, (9, 9), 0)
    ppm = meta["px_per_mm"]
    k = int(round(SQUARE_MM * ppm * 1.3))
    if k % 2 == 0:
        k += 1
    bg = cv2.medianBlur(gray, k)
    anomaly = bg.astype(int) - gray.astype(int)
    mask = (anomaly > ambang_anomali).astype(np.uint8) * 255

    Ox, Oy = mm_ke_px_ortho(np.asarray(O_mm, dtype=float), meta)
    Ox, Oy = int(round(Ox)), int(round(Oy))
    Yg, Xg = np.mgrid[0:mask.shape[0], 0:mask.shape[1]]
    dist_px = np.hypot(Xg - Ox, Yg - Oy)
    mask[dist_px > radius_cari_mm * ppm] = 0
    # Singkirkan kerucut citra gnomon (perkiraan geometris) — gnomon gelap
    # polos jangan sampai terbaca sebagai bayangan.
    geo = arah_citra_gnomon(meta, O_mm)
    if geo is not None:
        th_px = np.degrees(np.arctan2(Xg - Ox, -(Yg - Oy))) % 360
        beda = np.abs((th_px - geo[0] + 180) % 360 - 180)
        mask[(beda < geo[1]) & (dist_px > 3 * ppm)] = 0

    # Buang noise super kecil DULU (open), baru tutup celah tipis di objek
    # asli (close) — urutan ini penting: kalau closing dilakukan lebih dulu,
    # ia bisa menyambung noise-noise kecil yang tersebar jadi satu massa
    # besar sebelum sempat dibuang oleh opening.
    closed = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    terpilih = None
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 150:
            continue
        ys, xs = np.where(labels == i)
        dmin = float(np.hypot(xs - Ox, ys - Oy).min())
        if dmin < 15 * ppm and (terpilih is None or area > terpilih[1]):
            terpilih = (i, area)

    hasil = {"tip_mm": None, "tip_px": None, "yakin": False, "pakai_warna": False,
             "ujung_ambang_ketat": False, "n_piksel_gnomon": 0,
             "n_piksel_bayangan": 0, "mask_debug": closed}
    if terpilih is None:
        return hasil

    i, _ = terpilih
    ys, xs = np.where(labels == i)

    hsv = cv2.cvtColor(ortho_bgr, cv2.COLOR_BGR2HSV)
    warna_px = hsv[ys, xs]
    cocok_gnomon = cv2.inRange(warna_px.reshape(-1, 1, 3), warna_lo, warna_hi).reshape(-1) > 0
    n_gnomon = int(cocok_gnomon.sum())
    n_bayangan = int((~cocok_gnomon).sum())

    if n_gnomon > 0 and n_bayangan > 0:
        xs_pakai, ys_pakai = xs[~cocok_gnomon], ys[~cocok_gnomon]
        pakai_warna = True
    else:
        xs_pakai, ys_pakai = xs, ys
        pakai_warna = False

    if len(xs_pakai) == 0:
        return hasil

    # Lapis 3: dari kandidat bayangan, utamakan yang cukup gelap (bukan cuma
    # ekor penumbra pudar) untuk pencarian titik ujung.
    anomali_pakai = anomaly[ys_pakai, xs_pakai]
    tegas = anomali_pakai > ambang_ujung
    if tegas.any():
        xs_final, ys_final = xs_pakai[tegas], ys_pakai[tegas]
        ujung_ambang_ketat = True
    else:
        xs_final, ys_final = xs_pakai, ys_pakai
        ujung_ambang_ketat = False

    d = np.hypot(xs_final - Ox, ys_final - Oy)
    idx = int(np.argmax(d))
    tip_kasar = np.array([xs_final[idx], ys_final[idx]], dtype=float)

    # Lapis 4 (BARU): ARAH bayangan dari garis-fit lewat O, bukan dari satu
    # piksel terjauh. Piksel terjauh sering jatuh di TEPI bayangan (ujung
    # miring/penumbra, batas terang-teduh, putus di kotak hitam) sehingga
    # sudutnya melenceng beberapa derajat. Di sini semua piksel inti bayangan
    # di sisi ujung (dalam kerucut sekitar arah kasar, > 15 mm dari O agar
    # alas gnomon tak ikut) dipakai: arah = vektor eigen utama dari
    # sum(w * r * u u^T), w = kegelapan (anomali), u = vektor satuan dari O.
    # Dua iterasi: kerucut +-25 deg lalu +-10 deg. Panjang tetap dari
    # proyeksi titik terjauh ke garis itu.
    O_vec = np.array([Ox, Oy], dtype=float)
    arah = (tip_kasar - O_vec) / max(np.linalg.norm(tip_kasar - O_vec), 1e-9)
    P = np.stack([xs_final - Ox, ys_final - Oy], 1).astype(float)
    r = np.hypot(P[:, 0], P[:, 1])
    w_all = anomaly[ys_final, xs_final].astype(float)
    n_fit = 0
    for kerucut in (25.0, 10.0):
        with np.errstate(invalid="ignore", divide="ignore"):
            U = P / r[:, None]
        cosang = U @ arah
        sel = (r > 15 * ppm) & (cosang > math.cos(math.radians(kerucut)))
        if sel.sum() < 30:
            break
        w = w_all[sel] * r[sel]
        Us = U[sel]
        M = (Us * w[:, None]).T @ Us
        val, vec = np.linalg.eigh(M)
        baru = vec[:, int(np.argmax(val))]
        if baru @ arah < 0:
            baru = -baru
        arah = baru / np.linalg.norm(baru)
        n_fit = int(sel.sum())
    panjang = float((tip_kasar - O_vec) @ arah)
    tip_f = O_vec + arah * panjang
    tip_px = (int(round(tip_f[0])), int(round(tip_f[1])))
    tip_mm = px_ortho_ke_mm(tip_f, meta)
    hasil["n_piksel_fit"] = n_fit
    hasil["tip_kasar_px"] = (int(tip_kasar[0]), int(tip_kasar[1]))
    v0 = tip_kasar - O_vec
    hasil["koreksi_arah_deg"] = math.degrees(
        math.atan2(arah[1], arah[0]) - math.atan2(v0[1], v0[0]))

    hasil.update({
        "tip_mm": tip_mm, "tip_px": tip_px, "yakin": True,
        "pakai_warna": pakai_warna, "ujung_ambang_ketat": ujung_ambang_ketat,
        "n_piksel_gnomon": n_gnomon, "n_piksel_bayangan": n_bayangan,
        "mask_debug": closed,
    })
    return hasil
