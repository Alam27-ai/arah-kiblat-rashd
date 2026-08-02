"""
kalibrasi_kamera.py
===================
Kalibrasi kamera memakai papan ChArUco (papan_charuco_A4.pdf) untuk modul
Computer Vision BayangKiblat.

Kalibrasi WAJIB dilakukan sekali per kamera (per HP, per mode/lensa). Tanpa ini,
distorsi lensa membengkokkan garis lurus dan merusak pengukuran sudut.

Cara pakai
----------
1. Cetak papan_charuco_A4.pdf pada 100% ("Actual size"), tempel rata di alas kaku.
2. UKUR sisi satu kotak hitam dengan penggaris/jangka sorong.
   Isikan hasilnya ke SQUARE_MM di bawah (mis. 29.7 bila printer menyusut).
3. Ambil 15-25 foto papan dari berbagai sudut & jarak:
     - miring kiri/kanan/atas/bawah (~20-45 derajat), jangan tegak lurus semua
     - papan mengisi 1/3 sampai 2/3 bingkai
     - fokus tajam, cahaya merata, JANGAN pakai zoom digital
     - kunci fokus & jangan ganti mode kamera di tengah pemotretan
4. Simpan semua foto dalam satu folder, lalu jalankan:
       python kalibrasi_kamera.py folder_foto
5. Hasil tersimpan di kalibrasi_kamera.npz (dipakai tahap pengukuran).

Prasyarat:  pip install opencv-python numpy
"""

from __future__ import annotations

import glob
import os
import sys

import cv2
import numpy as np

# --- HARUS SESUAI PAPAN YANG DICETAK -----------------------------------------
SQUARE_MM = 30.0          # <-- GANTI dengan hasil UKUR sisi kotak (mm)
MARKER_MM = 22.0          # <-- rasio ikut tercetak; ukur juga bila ragu
NX, NY = 6, 7             # jumlah kotak (kolom, baris)
ARUCO_DICT = cv2.aruco.DICT_4X4_50
# -----------------------------------------------------------------------------

MIN_CORNERS = 8           # minimal sudut ChArUco agar satu foto dipakai
MIN_IMAGES = 8            # minimal foto valid agar kalibrasi layak


def build_board():
    adict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard(
        (NX, NY), SQUARE_MM / 1000.0, MARKER_MM / 1000.0, adict
    )
    return adict, board


def _make_detector(adict, board):
    """
    Kembalikan fungsi deteksi (gray -> (n, charuco_corners, charuco_ids)).
    OpenCV >= 4.8 memakai CharucoDetector; versi lama memakai API legacy.
    """
    params = cv2.aruco.DetectorParameters()
    # penyempurnaan sub-piksel: penting untuk akurasi
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

    if hasattr(cv2.aruco, "CharucoDetector"):
        ch_params = cv2.aruco.CharucoParameters()
        det = cv2.aruco.CharucoDetector(board, ch_params, params)

        def detect(gray):
            cc, ci, _, _ = det.detectBoard(gray)
            n = 0 if ci is None else len(ci)
            return n, cc, ci
        return detect

    # --- fallback OpenCV lama ---
    adet = cv2.aruco.ArucoDetector(adict, params)

    def detect_legacy(gray):
        mc, mids, _ = adet.detectMarkers(gray)
        if mids is None or len(mids) == 0:
            return 0, None, None
        n, cc, ci = cv2.aruco.interpolateCornersCharuco(mc, mids, gray, board)
        return (n or 0), cc, ci
    return detect_legacy


def collect(folder: str):
    """Deteksi sudut ChArUco pada semua foto di folder."""
    adict, board = build_board()
    detect = _make_detector(adict, board)

    exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG",
            "*.dng", "*.DNG", "*.heic", "*.HEIC")
    files: list[str] = []
    for e in exts:
        files.extend(glob.glob(os.path.join(folder, e)))
    files.sort()
    if not files:
        sys.exit(f"Tidak ada foto di: {folder}")

    all_corners, all_ids, size = [], [], None
    for f in files:
        img = cv2.imread(f)
        if img is None:
            print(f"  [lewati] tak terbaca: {os.path.basename(f)}")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]

        n, cc, ci = detect(gray)
        if n >= MIN_CORNERS:
            all_corners.append(cc)
            all_ids.append(ci)
            print(f"  [pakai]  {os.path.basename(f)}: {n} sudut")
        else:
            print(f"  [lewati] sudut terlalu sedikit ({n}): {os.path.basename(f)}")

    return all_corners, all_ids, size, board


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else "foto_kalibrasi"
    print(f"Papan: {NX}x{NY} kotak, sisi {SQUARE_MM} mm, marker {MARKER_MM} mm")
    print(f"Membaca foto dari: {folder}\n")

    corners, ids, size, board = collect(folder)
    print(f"\nFoto terpakai: {len(corners)}")
    if len(corners) < MIN_IMAGES:
        sys.exit(f"Kurang dari {MIN_IMAGES} foto valid — tambah foto dari sudut berbeda.")

    flags = cv2.CALIB_RATIONAL_MODEL          # model distorsi lebih lentur

    if hasattr(cv2.aruco, "calibrateCameraCharuco"):      # OpenCV lama
        rms, K, dist, _, _ = cv2.aruco.calibrateCameraCharuco(
            corners, ids, board, size, None, None, flags=flags
        )
    else:                                                  # OpenCV >= 4.9
        obj_pts, img_pts = [], []
        for cc, ci in zip(corners, ids):
            op, ip = board.matchImagePoints(cc, ci)
            if op is not None and len(op) >= 6:
                obj_pts.append(op)
                img_pts.append(ip)
        if len(obj_pts) < MIN_IMAGES:
            sys.exit("Pencocokan titik gagal pada terlalu banyak foto.")
        rms, K, dist, _, _ = cv2.calibrateCamera(
            obj_pts, img_pts, size, None, None, flags=flags
        )

    print("\n=== HASIL KALIBRASI ===")
    print(f"RMS reprojection error : {rms:.4f} piksel")
    print(f"fx, fy                 : {K[0,0]:.2f}, {K[1,1]:.2f}")
    print(f"cx, cy                 : {K[0,2]:.2f}, {K[1,2]:.2f}")
    print(f"Koefisien distorsi     : {np.ravel(dist)[:5]}")

    if rms < 0.5:
        print("Mutu: SANGAT BAIK (layak untuk pengukuran presisi).")
    elif rms < 1.0:
        print("Mutu: CUKUP. Bisa dipakai, tetapi foto lebih tajam akan membantu.")
    else:
        print("Mutu: KURANG. Ulangi: pastikan fokus tajam, papan rata, "
              "sudut pandang bervariasi, dan zoom digital dimatikan.")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "kalibrasi_kamera.npz")
    np.savez(out, K=K, dist=dist, rms=rms, image_size=size,
             square_mm=SQUARE_MM, marker_mm=MARKER_MM, nx=NX, ny=NY)
    print(f"\nTersimpan: {out}")


if __name__ == "__main__":
    main()
