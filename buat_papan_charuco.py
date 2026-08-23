#!/usr/bin/env python3
"""
buat_papan_charuco.py — Bangun papan_charuco_A4_v2.pdf: papan kalibrasi ChArUco
BayangKiblat + skala mm/cm di keempat tepi + titik gnomon tetap tercetak.

Pola ChArUco (ukuran, susunan, kamus) SENGAJA TIDAK DIUBAH dari versi awal —
kompatibel mundur, board.getChessboardCorners() tetap sama seperti sebelumnya.
Yang baru cuma dua: skala baca di tepi (untuk pemasangan fisik pasca-ukur,
lihat papan_charuco.titik_potong_tepi) dan titik gnomon tetap (menghapus satu
langkah klik di aplikasi, sekaligus jadi acuan baseline).

Kenapa skala per 1mm bukan 1cm: garis Kiblat hasil hitung nyaris tidak pernah
tepat jatuh di angka bulat. Goresan 1mm + baca desimal (dibantu interpolasi
mata atau crop-zoom di aplikasi) sudah cukup presisi -- lihat perhitungan
galat di RENCANA_CV.md.

Jalankan:  python buat_papan_charuco.py
Keluaran:  papan_charuco_A4_v2.pdf
"""

from __future__ import annotations

import io

import cv2
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

from papan_charuco import (
    ARUCO_DICT,
    BOARD_H_MM,
    BOARD_W_MM,
    GNOMON_TETAP_MM,
    MARKER_MM,
    NX,
    NY,
    SQUARE_MM,
    build_detector,
)

OUT = "papan_charuco_A4_v2.pdf"
PAGE_W, PAGE_H = 210.0, 297.0          # A4, mm
MARGIN_SISI = 15.0                     # mm, kiri/kanan (juga dipakai atas/bawah utk pola)
JUDUL_TINGGI = 12.0                    # mm, tinggi area judul di atas

# Posisi pola ChArUco di halaman (mm dari pojok kiri-atas halaman)
POLA_X = (PAGE_W - BOARD_W_MM) / 2.0                       # = 15.0
POLA_Y = JUDUL_TINGGI + MARGIN_SISI                         # = 27.0


def render_pola_charuco_png(px_per_mm: float = 12.0) -> bytes:
    """Render pola ChArUco (persis definisi di papan_charuco.py) sebagai PNG
    resolusi tinggi, siap ditempel presisi ke PDF."""
    _, board, _, _ = build_detector()
    w_px = int(BOARD_W_MM * px_per_mm)
    h_px = int(BOARD_H_MM * px_per_mm)
    img = board.generateImage((w_px, h_px))
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("Gagal render pola ChArUco ke PNG.")
    return buf.tobytes()


def _PX(x_dari_kiri_pola_mm: float) -> float:
    """Koordinat X halaman (satuan mm), dari X lokal pola (0=tepi kiri pola)."""
    return (POLA_X + x_dari_kiri_pola_mm) * mm


def _PY(y_dari_atas_pola_mm: float) -> float:
    """Koordinat Y halaman ReportLab (satuan mm, sumbu Y asli ReportLab
    mengarah KE ATAS dari dasar halaman), dari Y lokal pola (0=tepi ATAS
    pola, membesar ke bawah — sama seperti konvensi mm di papan_charuco.py).

    Dihitung manual per titik (bukan membalik seluruh kanvas dengan
    scale(1,-1)) supaya teks/angka TIDAK ikut tercermin terbalik.
    """
    return (PAGE_H - POLA_Y - y_dari_atas_pola_mm) * mm


def gambar_skala(c: canvas.Canvas, sisi: str):
    """Gambar goresan 1mm + label 1cm di satu sisi pola. `sisi` salah satu
    dari 'atas','bawah','kiri','kanan'. Semua koordinat dihitung lewat
    _PX/_PY supaya teks selalu tegak normal, tidak tercermin."""
    c.setLineWidth(0.25)
    c.setFont("Helvetica", 6)

    if sisi in ("atas", "bawah"):
        y0_lokal = 0.0 if sisi == "atas" else BOARD_H_MM
        arah = -1 if sisi == "atas" else 1   # goresan menjorok KELUAR pola
        for i in range(0, int(BOARD_W_MM) + 1):
            panjang = 2.5 if i % 10 == 0 else (1.6 if i % 5 == 0 else 1.0)
            x = _PX(i)
            c.line(x, _PY(y0_lokal), x, _PY(y0_lokal + arah * panjang))
            if i % 10 == 0:
                c.drawCentredString(x, _PY(y0_lokal + arah * (panjang + 3.2)),
                                    str(i // 10))

    else:  # kiri / kanan
        x0_lokal = 0.0 if sisi == "kiri" else BOARD_W_MM
        arah = -1 if sisi == "kiri" else 1
        for i in range(0, int(BOARD_H_MM) + 1):
            panjang = 2.5 if i % 10 == 0 else (1.6 if i % 5 == 0 else 1.0)
            y = _PY(i)
            c.line(_PX(x0_lokal), y, _PX(x0_lokal + arah * panjang), y)
            if i % 10 == 0:
                tx = _PX(x0_lokal + arah * (panjang + 4.5))
                c.drawCentredString(tx, y - 1.0 * mm, str(i // 10))


def gambar_titik_gnomon(c: canvas.Canvas):
    """Tandai titik pangkal gnomon TETAP dengan crosshair + label."""
    gx_lokal, gy_lokal = GNOMON_TETAP_MM
    gx, gy = _PX(gx_lokal), _PY(gy_lokal)
    r = 3.2 * mm
    c.setLineWidth(0.6)
    c.setStrokeColorRGB(0.75, 0.05, 0.05)
    c.circle(gx, gy, r, stroke=1, fill=0)
    c.line(gx - r - 1.5 * mm, gy, gx + r + 1.5 * mm, gy)
    c.line(gx, gy - r - 1.5 * mm, gx, gy + r + 1.5 * mm)

    # Label "O" dengan kotak putih di belakangnya -- posisi mendarat di atas
    # kotak ChArUco yang bisa hitam ATAU putih tergantung tempat O berada,
    # jadi teks polos saja bisa tak terbaca kalau jatuh di kotak hitam.
    c.setFont("Helvetica-Bold", 8)
    label_x, label_y = gx + r + 2.2 * mm, gy - 1.4 * mm
    lw = c.stringWidth("O", "Helvetica-Bold", 8)
    c.setFillColorRGB(1, 1, 1)
    c.rect(label_x - 0.6 * mm, label_y - 0.8 * mm, lw + 1.2 * mm, 3.6 * mm,
          stroke=0, fill=1)
    c.setFillColorRGB(0.75, 0.05, 0.05)
    c.drawString(label_x, label_y, "O")
    c.setFillColorRGB(0, 0, 0)
    c.setStrokeColorRGB(0, 0, 0)


def build():
    png_bytes = render_pola_charuco_png()
    img_reader = ImageReader(io.BytesIO(png_bytes))

    c = canvas.Canvas(OUT, pagesize=(PAGE_W * mm, PAGE_H * mm))

    # --- Judul ---
    c.setFont("Helvetica-Bold", 10)
    c.drawString(MARGIN_SISI * mm, (PAGE_H - 8) * mm,
                "Papan Kalibrasi ChArUco v2 — BayangKiblat")

    # --- Pola ChArUco (tak berubah dari versi awal) ---
    c.drawImage(img_reader, POLA_X * mm, (PAGE_H - POLA_Y - BOARD_H_MM) * mm,
               width=BOARD_W_MM * mm, height=BOARD_H_MM * mm)

    # _PX/_PY di dalam gambar_skala/gambar_titik_gnomon sudah mengonversi
    # koordinat "Y ke bawah dari atas pola" (konvensi papan_charuco.py) ke
    # koordinat asli ReportLab (Y ke atas dari dasar halaman) per titik —
    # jadi TIDAK perlu membalik kanvas di sini (itu tadi yang bikin teks
    # tercermin terbalik).
    for sisi in ("atas", "bawah", "kiri", "kanan"):
        gambar_skala(c, sisi)
    gambar_titik_gnomon(c)

    # --- Info teknis (koordinat normal reportlab, Y ke atas) ---
    info_y = PAGE_H - JUDUL_TINGGI - MARGIN_SISI - BOARD_H_MM - MARGIN_SISI - 6
    c.setFont("Helvetica", 7.2)
    baris = [
        f"Kamus ArUco: DICT_4X4_50    Susunan: {NX} x {NY} kotak",
        f"Sisi kotak: {SQUARE_MM:.1f} mm    Sisi marker: {MARKER_MM:.1f} mm    "
        f"Titik gnomon TETAP (O): {GNOMON_TETAP_MM[0]:.0f}mm, {GNOMON_TETAP_MM[1]:.0f}mm "
        "dari pojok kiri-atas pola",
        "",
        'CETAK 100% / "Actual size" — JANGAN "Fit to page".',
        "Setelah dicetak, UKUR sisi satu kotak hitam dengan penggaris/jangka sorong,",
        "lalu masukkan hasil ukur itu (bukan 30.0) ke aplikasi/skrip kalibrasi.",
        "Skala di tepi (mm/cm) juga ikut ukuran cetak — cek garis uji di bawah.",
        "Tempel rata pada alas kaku (triplek/MDF/foam board). Gunakan kertas doff.",
        "",
        "Letakkan PANGKAL GNOMON tepat di titik O bertanda silang merah di atas —",
        "posisi ini TETAP setiap kali dipakai, aplikasi tidak perlu Anda tandai lagi.",
    ]
    for i, b in enumerate(baris):
        c.drawString(MARGIN_SISI * mm, (info_y - i * 3.6) * mm, b)

    # --- Garis uji skala cetak (harus tepat 100mm bila cetak benar) ---
    uji_y = info_y - len(baris) * 3.6 - 6
    x0 = MARGIN_SISI
    c.setLineWidth(0.5)
    c.line(x0 * mm, uji_y * mm, (x0 + 100) * mm, uji_y * mm)
    for x in (x0, x0 + 100):
        c.line(x * mm, (uji_y - 1.5) * mm, x * mm, (uji_y + 1.5) * mm)
    c.setFont("Helvetica", 6.5)
    c.drawString(x0 * mm, (uji_y - 5) * mm,
                "Garis uji: panjang harus TEPAT 100 mm bila skala cetak benar.")

    c.showPage()
    c.save()
    print(f"Selesai: {OUT}")


if __name__ == "__main__":
    build()
