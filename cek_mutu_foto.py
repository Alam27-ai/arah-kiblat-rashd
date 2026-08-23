#!/usr/bin/env python3
"""
cek_mutu_foto.py — Alat kontrol mutu untuk foto papan ChArUco BayangKiblat.

Dipakai SEBELUM menjalankan kalibrasi_kamera.py, untuk menjawab pertanyaan:
"Apakah set foto yang baru saya ambil ini layak dipakai?"

Yang diperiksa (mode kalibrasi):
  1. Berapa sudut ChArUco terdeteksi per foto (dan apakah lolos ambang).
  2. Peta keterdeteksian per posisi sudut — menemukan area papan yang
     tertutup gnomon/bayangan/naungan secara SISTEMATIS (bukan acak), plus
     diagnosis otomatis penyebabnya (naungan vs distorsi vs cetakan rusak).
  3. Skala citra (mm per piksel) — indikator apakah foto sudah dikompresi
     atau papan terlalu jauh dari kamera.
  4. Statistik iluminasi di dalam area papan (highlight clipping, area gelap).
  5. Keragaman pose papan — penting agar kalibrasi terkendala dengan baik.
  6. Cakupan papan di SELURUH bidang frame (pusat vs tepi vs pojok) — koefisien
     distorsi hanya sahih di area yang pernah ditempati papan.

Mode ukur: menilai satu/beberapa foto PENGUKURAN (bukan set kalibrasi),
tanpa perlu data kalibrasi kamera sama sekali — lihat papan_charuco.cek_foto_ukur
untuk penjelasan prinsipnya.

Pemakaian:
    python cek_mutu_foto.py data_lapangan/sesi_2026-08-06
    python cek_mutu_foto.py data_lapangan/sesi_2026-08-06 --overlay hasil_overlay
    python cek_mutu_foto.py foto.jpg --mode ukur
    python cek_mutu_foto.py data_lapangan/sesi_hari_ini --mode ukur

Catatan: parameter papan (ukuran, susunan) ada di papan_charuco.py — satu
sumber kebenaran yang dipakai juga oleh kalibrasi_kamera.py dan app.py.
"""

from __future__ import annotations

import argparse
import collections
import glob
import os
import sys

import cv2
import numpy as np

from papan_charuco import (
    AMBANG_AMAN,
    AMBANG_ULANG,
    ARUCO_DICT,
    EXTS,
    GRID_H,
    GRID_W,
    MIN_CORNERS,
    NX,
    NY,
    SQUARE_MM,
    TOTAL_CORNERS,
    build_detector,
    cek_foto_ukur,
    illumination_stats,
    scale_mm_per_px,
)


def _diagnosa_marker(files, board, det):
    """Kenapa marker tertentu gagal terdeteksi? Bandingkan iluminasi tiap
    kotak marker antara yang sering gagal dan yang sehat.

    Ini memisahkan dua penyebab yang mudah tertukar:
      - marker TERANG dengan gradien besar -> tepi bayangan/berkas sinar
        melintasi marker (masalah LOKASI papan, bukan gnomon)
      - marker GELAP merata               -> papan ternaung, kurang cahaya
    Bayangan gnomon yang melintasi marker umumnya TIDAK mematikannya, jadi
    jangan buru-buru menyalahkan posisi gnomon.
    """
    adict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    mdet = cv2.aruco.ArucoDetector(adict, det.getDetectorParameters()
                                   if hasattr(det, "getDetectorParameters")
                                   else cv2.aruco.DetectorParameters())
    n_marker = NX * NY // 2
    obj_corners = board.getChessboardCorners()[:, :2]
    mk = board.getObjPoints()

    hadir = collections.Counter()
    terang = collections.defaultdict(list)
    rentang = collections.defaultdict(list)
    n_foto = 0

    for f in files:
        img = cv2.imread(f)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cc, ci, _, _ = det.detectBoard(gray)
        if ci is None or len(ci) < 12:
            continue
        n_foto += 1
        _, mi, _ = mdet.detectMarkers(gray)
        if mi is not None:
            hadir.update(mi.flatten().tolist())
        Hm, _ = cv2.findHomography(obj_corners[ci.flatten()].astype(np.float64),
                                   cc.reshape(-1, 2).astype(np.float64), 0)
        if Hm is None:
            continue
        gf = gray.astype(np.float32)
        for mid in range(n_marker):
            q = cv2.perspectiveTransform(
                mk[mid][:, :2].astype(np.float64).reshape(1, -1, 2), Hm).reshape(-1, 2)
            m = np.zeros(gray.shape, np.uint8)
            cv2.fillConvexPoly(m, q.astype(np.int32), 255)
            v = gf[m > 0]
            if v.size < 50:
                continue
            terang[mid].append(float(v.mean()))
            rentang[mid].append(float(np.percentile(v, 99) - np.percentile(v, 1)))

    if n_foto < 3:
        return
    gagal = [m for m in range(n_marker) if hadir.get(m, 0) < 0.5 * n_foto
             and terang.get(m)]
    if not gagal:
        print("\nDiagnosis marker: semua marker terbaca di mayoritas foto. Baik.")
        return

    sehat = [m for m in range(n_marker) if m not in gagal and terang.get(m)]
    t_gagal = float(np.mean([np.mean(terang[m]) for m in gagal]))
    t_sehat = float(np.mean([np.mean(terang[m]) for m in sehat])) if sehat else float("nan")
    r_gagal = float(np.mean([np.mean(rentang[m]) for m in gagal]))
    r_sehat = float(np.mean([np.mean(rentang[m]) for m in sehat])) if sehat else float("nan")

    print(f"\nDiagnosis marker — {len(gagal)} marker gagal di >50% foto: "
          f"{sorted(gagal)}")
    print(f"  kecerahan  gagal {t_gagal:6.1f}  vs  sehat {t_sehat:6.1f}")
    print(f"  rentang    gagal {r_gagal:6.1f}  vs  sehat {r_sehat:6.1f}")
    if t_gagal > t_sehat * 1.4 and r_gagal > r_sehat:
        print("  -> Marker itu JAUH LEBIH TERANG dengan gradien besar: ada tepi\n"
              "     bayangan / berkas sinar yang memotongnya. Pindahkan papan ke\n"
              "     area yang seluruhnya kena matahari langsung, tanpa naungan\n"
              "     sebagian. Ini masalah LOKASI papan, bukan posisi gnomon.")
    elif t_gagal < t_sehat * 0.7:
        print("  -> Marker itu jauh lebih gelap: area papan ternaung. "
              "Pindahkan ke tempat terbuka.")
    else:
        print("  -> Iluminasi mirip dengan marker sehat: kemungkinan cetakan\n"
              "     rusak/kotor, kertas terlipat, atau tertutup benda di area itu.")


def _peta_cakupan(all_corners, size, G: int = 6):
    """Seberapa merata sudut ChArUco menyebar di SELURUH bidang frame?

    Ini diagnosis terpenting untuk kalibrasi, dan yang paling sering
    terlewat. Distorsi radial paling kuat di tepi dan pojok citra. Kalau
    tidak ada satu pun titik data di sana, koefisien distorsi di wilayah
    itu hanyalah ekstrapolasi model — angkanya bisa tampak wajar tapi
    tidak punya dasar pengukuran sama sekali.
    """
    if not all_corners or size is None:
        return
    W, H = size
    hit = np.zeros((G, G), int)
    for cc in all_corners:
        for x, y in cc.reshape(-1, 2):
            i = min(int(y / H * G), G - 1)
            j = min(int(x / W * G), G - 1)
            hit[i, j] += 1

    print(f"\nCakupan frame oleh sudut ChArUco (frame dibagi {G}x{G} sel):")
    for i in range(G):
        print("   " + "".join(f"{hit[i, j]:5d}" for j in range(G)))
    kosong = int((hit == 0).sum())
    pojok = [hit[0, 0], hit[0, G - 1], hit[G - 1, 0], hit[G - 1, G - 1]]
    tepi = int((hit[0, :] == 0).sum() + (hit[-1, :] == 0).sum()
               + (hit[1:-1, 0] == 0).sum() + (hit[1:-1, -1] == 0).sum())
    print(f"  sel kosong: {kosong}/{G*G} ({100*kosong/G**2:.0f}%), "
          f"sel tepi kosong: {tepi}/{4*G-4}, "
          f"sel pojok terisi: {sum(1 for x in pojok if x > 0)}/4")
    if sum(1 for x in pojok if x > 0) < 3 or tepi > (4 * G - 4) // 2:
        print("  [!] Papan hampir tidak pernah menyentuh tepi/pojok frame.\n"
              "      Matriks kamera (K) akan baik, TAPI koefisien distorsi (D)\n"
              "      di tepi hanya ekstrapolasi dan tidak boleh dipercaya.\n"
              "      Ulangi dengan sengaja menggeser papan ke kiri, kanan, atas,\n"
              "      bawah, dan keempat pojok frame.")
    else:
        print("  Cakupan memadai — D terkendala oleh data nyata, bukan "
              "ekstrapolasi.")


def analyse(folder: str, overlay_dir: str | None = None):
    adict, board, params, det = build_detector()

    files: list[str] = []
    for e in EXTS:
        files.extend(glob.glob(os.path.join(folder, e)))
    files = sorted(set(files))
    if not files:
        sys.exit(f"Tidak ada foto di: {folder}")

    if overlay_dir:
        os.makedirs(overlay_dir, exist_ok=True)

    print(f"Papan: {NX}x{NY} kotak -> {TOTAL_CORNERS} sudut interior, "
          f"{NX * NY // 2} marker")
    print(f"Folder: {folder}  ({len(files)} foto)\n")
    print(f"{'foto':<26} {'sudut':>6} {'%':>4} {'mm/px':>7} "
          f"{'clip%':>6} {'gelap%':>7} {'kontras':>8} {'off-center':>11}")
    print("-" * 82)

    seen = collections.Counter()
    rows, all_corners, all_ids, size = [], [], [], None

    for f in files:
        img = cv2.imread(f)
        if img is None:
            print(f"{os.path.basename(f)[:26]:<26}  [tak terbaca]")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]
        h, w = gray.shape

        cc, ci, mc, mi = det.detectBoard(gray)
        n = 0 if ci is None else len(ci)
        if ci is not None:
            seen.update(ci.flatten().tolist())

        mmpx = scale_mm_per_px(cc, ci)
        clip, dark, contrast = illumination_stats(gray, cc)

        off = float("nan")
        if n >= 6:
            ctr = cc.reshape(-1, 2).mean(axis=0)
            off = float(np.hypot(*(ctr - np.array([w / 2, h / 2])))
                        / np.hypot(w / 2, h / 2))

        flag = "" if n >= MIN_CORNERS else "  <-- DITOLAK"
        print(f"{os.path.basename(f)[:26]:<26} {n:6d} {100*n/TOTAL_CORNERS:4.0f} "
              f"{mmpx:7.3f} {clip:6.1f} {dark:7.1f} {contrast:8.1f} "
              f"{off:11.2f}{flag}")

        rows.append((n, mmpx, clip, dark, off))
        if n >= MIN_CORNERS:
            all_corners.append(cc)
            all_ids.append(ci)

        if overlay_dir:
            vis = img.copy()
            if mi is not None:
                cv2.aruco.drawDetectedMarkers(vis, mc, mi)
            if cc is not None:
                for a, b in cc.reshape(-1, 2):
                    cv2.circle(vis, (int(a), int(b)), 5, (0, 0, 255), -1)
            cv2.putText(vis, f"{n}/{TOTAL_CORNERS} sudut", (10, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.imwrite(os.path.join(overlay_dir,
                                     "ov_" + os.path.basename(f) + ".png"), vis)

    print("-" * 82)
    ns = [r[0] for r in rows]
    mm = [r[1] for r in rows if not np.isnan(r[1])]
    lolos = sum(1 for x in ns if x >= MIN_CORNERS)
    print(f"Rata-rata sudut terdeteksi : {np.mean(ns):.1f}/{TOTAL_CORNERS} "
          f"({100*np.mean(ns)/TOTAL_CORNERS:.0f}%)")
    print(f"Foto lolos ambang ({MIN_CORNERS} sudut) : {lolos}/{len(ns)}")
    if mm:
        print(f"Skala rata-rata            : {np.mean(mm):.3f} mm/piksel "
              f"(1 kotak {SQUARE_MM:.0f} mm = {SQUARE_MM/np.mean(mm):.0f} px)")

    print(f"\nPeta keterdeteksian sudut ({GRID_W} kolom x {GRID_H} baris), "
          f"dari {len(files)} foto:")
    print("        " + "".join(f"kol{c}  " for c in range(GRID_W)))
    for r in range(GRID_H):
        print(f"baris{r} " + "".join(f"{seen.get(r*GRID_W+c, 0):4d}  "
                                     for c in range(GRID_W)))
    print("  (angka rendah yang MENGELOMPOK = kegagalan SISTEMATIS di satu area\n"
          "   papan, bukan kegagalan acak. Lihat diagnosis per marker di bawah\n"
          "   untuk mengetahui penyebabnya.)")

    _diagnosa_marker(files, board, det)
    _peta_cakupan(all_corners, size)

    if len(all_corners) >= 4:
        objp, imgp = [], []
        for cc, ci in zip(all_corners, all_ids):
            o, i2 = board.matchImagePoints(cc, ci)
            objp.append(o)
            imgp.append(i2)
        try:
            rms, K, D, rvecs, _ = cv2.calibrateCamera(objp, imgp, size, None, None)
            angs = []
            for rv in rvecs:
                R, _ = cv2.Rodrigues(rv)
                nrm = R @ np.array([0, 0, 1.0])
                angs.append(np.degrees(np.arccos(abs(np.clip(nrm[2], -1, 1)))))
            angs = np.array(angs)
            print(f"\nUji kalibrasi cepat ({len(objp)} foto):")
            print(f"  RMS reproyeksi        : {rms:.3f} piksel")
            print(f"  Kemiringan papan      : {angs.min():.1f}deg .. "
                  f"{angs.max():.1f}deg (rentang {np.ptp(angs):.1f}deg)")
            if np.ptp(angs) < 25:
                print("  [!] Rentang kemiringan sempit — parameter distorsi "
                      "akan lemah terkendala.\n      Ambil foto dari lebih "
                      "banyak sudut miring yang berbeda.")
        except cv2.error as exc:
            print(f"\n[!] Kalibrasi cepat gagal: {exc}")

    print("\nPeringatan:")
    warn = False
    if mm and np.mean(mm) > 0.4:
        warn = True
        print(f"  [!] Skala kasar ({np.mean(mm):.2f} mm/px). Foto kemungkinan "
              "dikompresi (WhatsApp/Telegram)\n      atau papan terlalu jauh. "
              "Pakai file asli dari kamera, dekatkan papan.")
    bad_dark = [i for i, r in enumerate(rows) if r[3] > 25]
    if bad_dark:
        warn = True
        print(f"  [!] {len(bad_dark)} foto punya >25% area papan yang gelap "
              "(naungan sebagian).\n      Ambil foto di area terbuka penuh.")
    bad_clip = [i for i, r in enumerate(rows) if r[2] > 2]
    if bad_clip:
        warn = True
        print(f"  [!] {len(bad_clip)} foto punya highlight jenuh >2%. "
              "Gunakan kertas doff / kurangi eksposur.")
    far = [i for i, r in enumerate(rows) if r[4] > 0.5]
    if far:
        warn = True
        print(f"  [!] {len(far)} foto menempatkan papan jauh dari pusat frame. "
              "Distorsi lensa\n      paling besar di tepi — usahakan papan "
              "di tengah frame.")
    if lolos < 8:
        warn = True
        print(f"  [!] Hanya {lolos} foto lolos; kalibrasi_kamera.py butuh "
              "minimal 8.")
    if not warn:
        print("  (tidak ada) — set foto ini layak untuk kalibrasi.")


def periksa_ukur(target: str) -> None:
    if os.path.isdir(target):
        files: list[str] = []
        for e in EXTS:
            files.extend(glob.glob(os.path.join(target, e)))
        files = sorted(set(files))
    else:
        files = [target]
    if not files:
        sys.exit(f"Tidak ada foto di: {target}")

    print("MODE UKUR — cek mutu foto pengukuran tanpa data kalibrasi\n"
          f"Ambang: AMAN < {AMBANG_AMAN} deg, ULANGI > {AMBANG_ULANG} deg\n")
    hasil = [cek_foto_ukur(f, verbose=True) for f in files]
    tally = collections.Counter(h["vonis"] for h in hasil)
    print("-" * 60)
    print(f"Ringkasan: {tally['AMAN']} AMAN, {tally['SEDANG']} SEDANG, "
          f"{tally['ULANGI']} ULANGI  (dari {len(hasil)} foto)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", help="folder foto (mode kalibrasi) "
                                   "atau folder/berkas (mode ukur)")
    ap.add_argument("--mode", choices=("kalibrasi", "ukur"), default="kalibrasi",
                    help="kalibrasi: nilai satu SET foto untuk kalibrasi kamera. "
                         "ukur: nilai foto pengukuran satu per satu, tanpa "
                         "perlu data kalibrasi.")
    ap.add_argument("--overlay", metavar="DIR", default=None,
                    help="tulis citra overlay deteksi ke DIR untuk cek visual")
    a = ap.parse_args()
    if a.mode == "ukur":
        periksa_ukur(a.target)
    else:
        analyse(a.target, a.overlay)


if __name__ == "__main__":
    main()
