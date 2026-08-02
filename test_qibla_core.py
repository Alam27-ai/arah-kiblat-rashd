"""
test_qibla_core.py
==================
Validasi engine. Jalankan: python test_qibla_core.py
(Membutuhkan koneksi internet sekali untuk mengunduh de440s.bsp.)
"""

import datetime as dt

import qibla_core as qc


def test_azimuth_kiblat_arah_barat_laut():
    """Untuk lokasi di Indonesia, Kiblat ada di barat laut (± 290°–295°)."""
    a_k = qc.azimuth_kiblat(-6.98, 109.72)  # Pekalongan
    assert 285.0 < a_k < 300.0, f"A_k di luar dugaan: {a_k}"
    print(f"[OK] A_k Pekalongan = {a_k:.4f}°")


def test_target_azimuth_konsistensi():
    """Cek turunan azimuth bayangan sesuai desain (pagi kanan, sore kiri)."""
    a_k = 295.0
    a_s_pagi = qc.target_sun_azimuth(a_k, 30, "pagi")
    assert abs((a_s_pagi + 180) % 360 - (a_k - 30) % 360) < 1e-9
    a_s_sore = qc.target_sun_azimuth(a_k, 120, "sore")
    assert abs((a_s_sore + 180) % 360 - (a_k + 120) % 360) < 1e-9
    print(f"[OK] A_s pagi={a_s_pagi:.3f}, sore={a_s_sore:.3f}")


def test_solver_menemukan_dan_konsisten():
    """Solver menemukan waktu dan azimuth Matahari saat itu == A_s (±0.01°)."""
    lat, lon, elev = -6.98, 109.72, 5.0
    the_date = dt.date(2026, 3, 21)  # ekuinoks, kondisi 'jinak'
    a_k = qc.azimuth_kiblat(lat, lon)
    a_s = qc.target_sun_azimuth(a_k, 30, "pagi")
    sols = qc.solve(lat, lon, elev, the_date, "pagi", a_s, 7.0, height=1.0)
    assert sols, "Solver gagal menemukan waktu pada kasus jinak"
    for s in sols:
        err = abs(qc._ang_diff(s.sun_azimuth, a_s))
        assert err < 0.01, f"Azimuth solusi meleset {err}° dari A_s"
        # panjang bayangan konsisten: tinggi/tan(alt)
        import math
        expected = 1.0 / math.tan(math.radians(s.sun_altitude))
        assert abs(s.shadow_length - expected) < 1e-6
    print(f"[OK] Solver: {sols[0].time_str()} WIB, "
          f"az={sols[0].sun_azimuth:.4f}° (target {a_s:.4f}°), "
          f"alt={sols[0].sun_altitude:.2f}°, bayangan={sols[0].shadow_length:.3f} m")


def test_kasus_tanpa_solusi():
    """ΔA ekstrem bisa membuat target tak terlintasi -> daftar kosong, tanpa error."""
    lat, lon, elev = -6.98, 109.72, 5.0
    the_date = dt.date(2026, 3, 21)
    a_k = qc.azimuth_kiblat(lat, lon)
    a_s = qc.target_sun_azimuth(a_k, 5, "pagi")  # target mepet, mungkin tak terlintas pagi
    sols = qc.solve(lat, lon, elev, the_date, "pagi", a_s, 7.0)
    print(f"[OK] Kasus ΔA=5 pagi -> {len(sols)} solusi (tanpa crash)")


def test_compute_all_terurut_dan_bulat():
    """compute_all: terurut waktu, ΔA bulat, geometri busur→Kiblat konsisten."""
    lat = qc.dms_to_decimal(6, 53, 18.96, "S")
    lon = qc.dms_to_decimal(109, 40, 31.08, "E")
    res = qc.compute_all(lat, lon, 0.0, dt.date(2026, 7, 28), 7.0, 1.0, step=5)
    rows, a_k = res["rows"], res["a_k"]
    assert rows, "compute_all mengembalikan daftar kosong"
    assert rows == sorted(rows, key=lambda x: x.local_time), "tidak terurut waktu"
    for s in rows:
        assert float(s.delta_a).is_integer() and s.delta_a % 5 == 0, "ΔA tidak bulat"
        q = (s.shadow_azimuth + (s.delta_a if s.session == "pagi" else -s.delta_a)) % 360
        assert abs(qc._ang_diff(q, a_k)) < 0.1, "geometri busur→Kiblat meleset"
    print(f"[OK] compute_all: {len(rows)} waktu, ΔA bulat, geometri konsisten")


def test_rashdul_geometri():
    """rashdul_qiblah: 'searah' -> azimuth=A_k; 'sebaliknya' -> azimuth=A_k+180."""
    lat = qc.dms_to_decimal(6, 53, 18.96, "S")
    lon = qc.dms_to_decimal(109, 40, 31.08, "E")
    r = qc.rashdul_qiblah(lat, lon, 0.0, dt.date(2026, 1, 15), 7.0, 1.0)
    a_k = r["a_k"]
    assert r["events"], "tidak ada event Rashdul pada tanggal uji"
    for e in r["events"]:
        s = e["solution"]
        tgt = a_k if e["kind"] == "searah" else (a_k + 180.0) % 360.0
        assert abs(qc._ang_diff(s.sun_azimuth, tgt)) < 0.05, "azimuth event meleset dari target"
    print(f"[OK] rashdul: {len(r['events'])} event, azimuth = target")


def test_solve_instant_geometri_dan_arah():
    """solve_instant: sudut putar tak dibulatkan, arah kanan/kiri konsisten dgn tanda selisih."""
    lat = qc.dms_to_decimal(6, 53, 18.96, "S")
    lon = qc.dms_to_decimal(109, 40, 31.08, "E")
    tz = dt.timezone(dt.timedelta(hours=7))
    a_k = qc.azimuth_kiblat(lat, lon)

    for hh in (7, 9, 12, 15, 17):
        dt_local = dt.datetime(2026, 7, 30, hh, 13, 27, tzinfo=tz)
        out = qc.solve_instant(lat, lon, 0.0, dt_local, height=1.0)
        sol, arah = out["solution"], out["arah"]
        assert out["a_k"] == a_k

        if sol.sun_altitude <= 0:
            continue  # matahari di bawah ufuk, lewati

        # delta_a umumnya TIDAK bulat (inti fitur mode bebas waktu)
        assert sol.delta_a >= 0.0
        assert sol.shadow_azimuth == (sol.sun_azimuth + 180.0) % 360.0

        # geometri: memutar shadow_azimuth ke arah yg benar sejauh delta_a harus == a_k
        sign = 1.0 if arah == "kanan" else -1.0
        hasil = (sol.shadow_azimuth + sign * sol.delta_a) % 360.0
        assert abs(qc._ang_diff(hasil, a_k)) < 1e-6, (
            f"jam {hh}: arah={arah} delta={sol.delta_a} tidak sampai ke A_k"
        )

    # waktu tanpa tzinfo harus ditolak
    try:
        qc.solve_instant(lat, lon, 0.0, dt.datetime(2026, 7, 30, 9, 0, 0))
        assert False, "seharusnya ValueError untuk datetime naive"
    except ValueError:
        pass
    print("[OK] solve_instant: geometri & arah konsisten di semua jam uji")


def test_dms_bolak_balik():
    """Konversi desimal↔DMS konsisten (contoh koordinat Kakbah)."""
    s = qc.decimal_to_dms(qc.KAABA_LAT, "lat")
    assert s.startswith("21° 25' 21.17"), f"DMS Kakbah tak sesuai: {s}"
    back = qc.dms_to_decimal(21, 25, 21.17, "N")
    assert abs(back - qc.KAABA_LAT) < 1e-5
    print(f"[OK] DMS Kakbah = {s}")


if __name__ == "__main__":
    test_azimuth_kiblat_arah_barat_laut()
    test_target_azimuth_konsistensi()
    test_solver_menemukan_dan_konsisten()
    test_kasus_tanpa_solusi()
    test_compute_all_terurut_dan_bulat()
    test_rashdul_geometri()
    test_solve_instant_geometri_dan_arah()
    test_dms_bolak_balik()
    print("\nSemua pengujian selesai.")
