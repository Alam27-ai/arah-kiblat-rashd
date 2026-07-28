"""
qibla_core.py
=============
Engine perhitungan Arah Kiblat berbasis metode
"Selisih Azimuth Matahari Harian" (Rashdul Qiblah Harian).

Tidak bergantung pada Streamlit sehingga mudah diuji terpisah.

Konvensi azimuth: dari Utara sejati, searah jarum jam (0°..360°),
konsisten dengan output .altaz() Skyfield.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, date as date_cls

from skyfield.api import load, wgs84

# ---------------------------------------------------------------------------
# Koordinat Kakbah (DMS acuan resmi Kemenag RI)
#   Lintang : 21° 25' 21.17" N  => 21.422547222°
#   Bujur   : 39° 49' 34.56" E  => 39.826266667°
# ---------------------------------------------------------------------------
KAABA_LAT = 21.422547222
KAABA_LON = 39.826266667

EPHEMERIS_NAME = "de440s.bsp"  # versi ringkas ~32 MB (rentang 1849-2150)

_EPH = None
_TS = None


# ---------------------------------------------------------------------------
# EPHEMERIS
# ---------------------------------------------------------------------------
def load_ephemeris(name: str = EPHEMERIS_NAME):
    """Muat kernel ephemeris JPL + timescale. Unduh otomatis bila belum ada."""
    global _EPH, _TS
    if _EPH is None:
        _EPH = load(name)
        _TS = load.timescale()
    return _EPH, _TS


# ---------------------------------------------------------------------------
# KONVERSI DERAJAT-MENIT-DETIK (DMS)
# ---------------------------------------------------------------------------
def dms_to_decimal(deg: float, minute: float, sec: float, hemisphere: str = "N") -> float:
    """Ubah DMS + arah mata angin (N/S/E/W) menjadi derajat desimal bertanda."""
    val = abs(deg) + minute / 60.0 + sec / 3600.0
    if hemisphere.upper() in ("S", "W"):
        val = -val
    return val


def decimal_to_dms(value: float, kind: str = "az", sec_dp: int = 2) -> str:
    """
    Format derajat desimal -> string DMS.
      kind='lat' -> tambahkan N/S       (mis. 6° 59' 00.00" S)
      kind='lon' -> tambahkan E/W
      kind='az'  -> azimuth 0..360, derajat 3 digit (mis. 294° 39' 57.00")
      kind='alt' -> altitude/derajat bebas tanda
    """
    hemi = ""
    if kind == "lat":
        hemi = " N" if value >= 0 else " S"
        v = abs(value)
    elif kind == "lon":
        hemi = " E" if value >= 0 else " W"
        v = abs(value)
    elif kind == "az":
        v = value % 360.0
    else:  # 'alt' atau sudut umum
        v = abs(value)

    d = int(v)
    m_full = (v - d) * 60.0
    m = int(m_full)
    s = round((m_full - m) * 60.0, sec_dp)
    # koreksi pembulatan detik/menit
    if s >= 60.0:
        s -= 60.0
        m += 1
    if m >= 60:
        m -= 60
        d += 1

    sign = "-" if (kind == "alt" and value < 0) else ""
    width = 3 + sec_dp if sec_dp > 0 else 2
    sec_str = f"{s:0{width}.{sec_dp}f}"
    if kind == "az":
        return f"{d:03d}° {m:02d}' {sec_str}\"{hemi}"
    return f"{sign}{d}° {m:02d}' {sec_str}\"{hemi}"


# ---------------------------------------------------------------------------
# AZIMUTH KIBLAT (Great Circle / initial bearing)
# ---------------------------------------------------------------------------
def azimuth_kiblat(lat: float, lon: float) -> float:
    """Azimuth Kiblat (0..360) dari Utara searah jarum jam."""
    phi1 = math.radians(lat)
    phi2 = math.radians(KAABA_LAT)
    dlon = math.radians(KAABA_LON - lon)
    x = math.sin(dlon)
    y = math.cos(phi1) * math.tan(phi2) - math.sin(phi1) * math.cos(dlon)
    return math.degrees(math.atan2(x, y)) % 360.0


# ---------------------------------------------------------------------------
# TARGET AZIMUTH MATAHARI (A_s)
# ---------------------------------------------------------------------------
def target_sun_azimuth(a_k: float, delta_a: float, session: str) -> float:
    """
    Pagi : A_s = (A_k - 180 - ΔA) mod 360  -> geser busur ke KANAN
    Sore : A_s = (A_k + ΔA - 180) mod 360  -> geser busur ke KIRI
    """
    session = session.lower()
    if session == "pagi":
        return (a_k - 180.0 - delta_a) % 360.0
    if session == "sore":
        return (a_k + delta_a - 180.0) % 360.0
    raise ValueError("session harus 'pagi' atau 'sore'")


# ---------------------------------------------------------------------------
# UTILITAS SUDUT & POSISI MATAHARI
# ---------------------------------------------------------------------------
def _ang_diff(a: float, b: float) -> float:
    """Selisih a-b ter-wrap ke (-180, 180]."""
    return (a - b + 180.0) % 360.0 - 180.0


def _make_observer(lat: float, lon: float, elev: float):
    eph, ts = load_ephemeris()
    observer = eph["earth"] + wgs84.latlon(lat, lon, elevation_m=elev)
    return observer, eph["sun"], ts


def sun_altaz(observer, sun, ts, dt_utc: datetime):
    """Kembalikan (altitude_deg, azimuth_deg) Matahari (apparent) pada waktu UTC."""
    t = ts.from_datetime(dt_utc)
    app = observer.at(t).observe(sun).apparent()
    alt, az, _ = app.altaz()
    return alt.degrees, az.degrees


def shadow_length(height: float, altitude_deg: float):
    """Panjang bayangan tiang tegak = tinggi / tan(altitude). None bila Matahari terlalu rendah."""
    if altitude_deg <= 0.5:
        return None
    return height / math.tan(math.radians(altitude_deg))


# ---------------------------------------------------------------------------
# HASIL SOLVER
# ---------------------------------------------------------------------------
@dataclass
class Solution:
    local_time: datetime
    sun_azimuth: float
    sun_altitude: float
    shadow_length: float | None
    shadow_azimuth: float
    delta_a: float | None = None
    session: str = ""

    def time_str(self) -> str:
        return self.local_time.strftime("%H:%M:%S")


# ---------------------------------------------------------------------------
# SCANNER (dibuat sekali, dipakai ulang untuk semua ΔA)
# ---------------------------------------------------------------------------
def _build_scanner(lat, lon, elev, the_date, session, tz_offset):
    """Kembalikan (local_dt, altaz_at, start_h, end_h) untuk sesi terkait."""
    observer, sun, ts = _make_observer(lat, lon, elev)
    tz = timezone(timedelta(hours=tz_offset))
    start_h, end_h = (6.0, 11.5) if session.lower() == "pagi" else (13.0, 17.5)

    def local_dt(hour_float: float) -> datetime:
        base = datetime(the_date.year, the_date.month, the_date.day, tzinfo=tz)
        return base + timedelta(hours=hour_float)

    def altaz_at(hour_float: float):
        dt_utc = local_dt(hour_float).astimezone(timezone.utc)
        return sun_altaz(observer, sun, ts, dt_utc)  # (alt, az)

    return local_dt, altaz_at, start_h, end_h


def _scan_samples(altaz_at, start_h, end_h, step_seconds: float):
    """Pemindaian kasar seluruh window: list (hour, alt, az)."""
    step = step_seconds / 3600.0
    out = []
    h = start_h
    while h <= end_h + 1e-9:
        alt, az = altaz_at(h)
        out.append((h, alt, az))
        h += step
    return out


def _find_roots(samples, a_s, altaz_at):
    """Cari jam saat azimuth == A_s dari sampel (deteksi ganti tanda -> bisection)."""
    roots = []
    for i in range(1, len(samples)):
        h0, _, az0 = samples[i - 1]
        h1, _, az1 = samples[i]
        d0 = _ang_diff(az0, a_s)
        d1 = _ang_diff(az1, a_s)
        if d0 == 0.0:
            roots.append(h0)
            continue
        # crossing asli: ganti tanda dengan magnitudo kecil di kedua sisi
        if d0 * d1 < 0 and abs(d0) < 90 and abs(d1) < 90:
            lo, hi, flo = h0, h1, d0
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                _, azm = altaz_at(mid)
                fmid = _ang_diff(azm, a_s)
                if abs(hi - lo) < 1e-7:
                    break
                if flo * fmid <= 0:
                    hi = mid
                else:
                    lo, flo = mid, fmid
            roots.append(0.5 * (lo + hi))
    return roots


def _make_solution(local_dt, altaz_at, hour, height, delta_a, session) -> Solution:
    dt_local = local_dt(hour)
    alt, az = altaz_at(hour)
    return Solution(
        local_time=dt_local,
        sun_azimuth=az,
        sun_altitude=alt,
        shadow_length=shadow_length(height, alt),
        shadow_azimuth=(az + 180.0) % 360.0,
        delta_a=delta_a,
        session=session,
    )


# ---------------------------------------------------------------------------
# SOLVE (satu A_s) — dipertahankan untuk kompatibilitas & pengujian
# ---------------------------------------------------------------------------
def solve(lat, lon, elev, the_date, session, a_s, tz_offset,
          height: float = 1.0, step_seconds: float = 20.0) -> list[Solution]:
    local_dt, altaz_at, start_h, end_h = _build_scanner(lat, lon, elev, the_date, session, tz_offset)
    samples = _scan_samples(altaz_at, start_h, end_h, step_seconds)
    roots = _find_roots(samples, a_s, altaz_at)
    return [_make_solution(local_dt, altaz_at, r, height, None, session) for r in roots]


# ---------------------------------------------------------------------------
# HITUNG SEMUA PELUANG (berbasis WAKTU; pagi + sore digabung, terurut waktu)
# ---------------------------------------------------------------------------
def compute_all(lat, lon, elev, the_date, tz_offset, height: float = 1.0,
                step: int = 5, alt_min: float = 8.0, alt_max: float = 70.0,
                step_seconds: float = 20.0) -> dict:
    """
    Enumerasi SEMUA peluang eksekusi pada tanggal ini berdasarkan ΔA BULAT
    (kelipatan `step` derajat) pada rentang penuh 1°..179°, untuk kedua sesi.
    Untuk tiap ΔA dicari WAKTU tepat (detik) saat azimuth Matahari = target,
    lalu disaring hanya yang tinggi Mataharinya dalam [alt_min, alt_max].

    Keunggulan: sudut busur selalu bulat (mudah & akurat disetel), sekaligus
    menangkap peluang di segala musim (rentang penuh + saringan altitude).
    Waktu boleh "ganjil" — ditangani oleh hitung mundur.

    Kembalikan {a_k, rows} — rows = daftar Solution terurut menurut waktu.
    """
    a_k = azimuth_kiblat(lat, lon)
    rows: list[Solution] = []
    for session in ("pagi", "sore"):
        local_dt, altaz_at, s_h, e_h = _build_scanner(
            lat, lon, elev, the_date, session, tz_offset)
        samples = _scan_samples(altaz_at, s_h, e_h, step_seconds)
        for da in range(step, 180, step):          # 5,10,… atau 1,2,… (kelipatan step)
            a_s = target_sun_azimuth(a_k, da, session)
            for r in _find_roots(samples, a_s, altaz_at):
                sol = _make_solution(local_dt, altaz_at, r, height, da, session)
                if alt_min <= sol.sun_altitude <= alt_max:
                    rows.append(sol)
    rows.sort(key=lambda x: x.local_time)
    return {"a_k": a_k, "rows": rows}


# ---------------------------------------------------------------------------
# RASHDUL QIBLAH HARIAN (kasus ΔA = 0, tanpa busur)
# ---------------------------------------------------------------------------
def rashdul_qiblah(lat, lon, elev, the_date, tz_offset, height: float = 1.0,
                   start_h: float = 5.5, end_h: float = 18.5,
                   step_seconds: float = 20.0) -> dict:
    """
    Cari saat-saat istimewa pada tanggal ini ketika bayangan tongkat LANGSUNG
    menunjukkan garis Kiblat tanpa perlu busur (ΔA = 0):

      - kind="searah"     : azimuth Matahari = A_k (Matahari tepat searah Kiblat).
                            Bayangan menunjuk berlawanan Kiblat; Kiblat = arah
                            dari ujung bayangan (B) menuju pangkal (O) lalu ke Matahari.
      - kind="sebaliknya" : azimuth Matahari = A_k + 180° (Matahari membelakangi
                            Kiblat). Bayangan menunjuk TEPAT ke Kiblat; garis O→B
                            adalah arah Kiblat.

    Dipindai sepanjang siang (start_h..end_h). Hanya kejadian dengan Matahari di
    atas ufuk yang dikembalikan. Bisa 0, 1, atau 2 kejadian per jenis per hari.

    Kembalikan {a_k, events} — events = daftar {kind, solution} terurut waktu.
    """
    a_k = azimuth_kiblat(lat, lon)
    observer, sun, ts = _make_observer(lat, lon, elev)
    tz = timezone(timedelta(hours=tz_offset))

    def local_dt(h):
        base = datetime(the_date.year, the_date.month, the_date.day, tzinfo=tz)
        return base + timedelta(hours=h)

    def altaz_at(h):
        return sun_altaz(observer, sun, ts, local_dt(h).astimezone(timezone.utc))

    samples = _scan_samples(altaz_at, start_h, end_h, step_seconds)

    events = []
    for kind, target in (("searah", a_k), ("sebaliknya", (a_k + 180.0) % 360.0)):
        for r in _find_roots(samples, target, altaz_at):
            alt, az = altaz_at(r)
            if alt <= 0.0:          # Matahari di bawah ufuk → tak terpakai
                continue
            events.append({
                "kind": kind,
                "solution": Solution(
                    local_time=local_dt(r), sun_azimuth=az, sun_altitude=alt,
                    shadow_length=shadow_length(height, alt),
                    shadow_azimuth=(az + 180.0) % 360.0, delta_a=0.0, session=kind,
                ),
            })
    events.sort(key=lambda e: e["solution"].local_time)
    return {"a_k": a_k, "events": events}
