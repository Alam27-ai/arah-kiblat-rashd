"""
app.py
======
Aplikasi Streamlit — Penentu Arah Kiblat metode
"Selisih Azimuth Matahari Harian" (Rashdul Qiblah Harian).

Alur baru (lebih sederhana):
  Lokasi + Tanggal  ->  satu tombol "Hitung"  ->  daftar SEMUA waktu peluang
  hari itu (pagi & sore digabung, terurut waktu). Pengguna memilih satu waktu
  (default: yang terdekat), lalu simulasi + hitung mundur mengikuti pilihan itu.

Jalankan:  streamlit run app.py
Perhitungan astronomi ada di qibla_core.py.
"""

from __future__ import annotations

import datetime as dt
import math
import re
import tempfile
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from streamlit_js_eval import get_geolocation

import qibla_core as qc

# Nama bulan Indonesia (tidak bergantung locale server)
BULAN_ID = ["", "Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli",
            "Agustus", "September", "Oktober", "November", "Desember"]


def tgl_id(d):
    """Format tanggal Indonesia, mis. '28 Juli 2026'."""
    return f"{d.day:02d} {BULAN_ID[d.month]} {d.year}"


# ---------------------------------------------------------------------------
st.set_page_config(page_title="BayangKiblat — Arah Kiblat via Bayangan Matahari",
                   page_icon="🕋", layout="wide")

TZ_OPTIONS = {"WIB (UTC+7)": 7.0, "WITA (UTC+8)": 8.0, "WIT (UTC+9)": 9.0, "Kustom": None}
OFF2LABEL = {7.0: "WIB (UTC+7)", 8.0: "WITA (UTC+8)", 9.0: "WIT (UTC+9)"}
MANUAL_OPT = "📍 Manual / isi sendiri"

# Rentang ΔA per sesi (dipakai internal; pengguna tak perlu memilih).
RANGE_PAGI = (10, 80)
RANGE_SORE = (100, 170)

# Daftar kota Indonesia: nama -> (lintang, bujur, offset UTC)
CITIES = {
    "Banda Aceh": (5.5483, 95.3238, 7.0), "Medan": (3.5952, 98.6722, 7.0),
    "Padang": (-0.9471, 100.4172, 7.0), "Pekanbaru": (0.5071, 101.4478, 7.0),
    "Palembang": (-2.9761, 104.7754, 7.0), "Bandar Lampung": (-5.3971, 105.2668, 7.0),
    "Jakarta": (-6.2088, 106.8456, 7.0), "Bogor": (-6.5971, 106.8060, 7.0),
    "Bandung": (-6.9175, 107.6191, 7.0), "Pekalongan": (-6.8886, 109.6753, 7.0),
    "Semarang": (-6.9932, 110.4203, 7.0), "Surakarta (Solo)": (-7.5755, 110.8243, 7.0),
    "Yogyakarta": (-7.7956, 110.3695, 7.0), "Surabaya": (-7.2575, 112.7521, 7.0),
    "Malang": (-7.9666, 112.6326, 7.0), "Denpasar": (-8.6705, 115.2126, 8.0),
    "Mataram": (-8.5833, 116.1167, 8.0), "Banjarmasin": (-3.3194, 114.5908, 8.0),
    "Balikpapan": (-1.2379, 116.8529, 8.0), "Samarinda": (-0.5017, 117.1536, 8.0),
    "Makassar": (-5.1477, 119.4327, 8.0), "Manado": (1.4748, 124.8421, 8.0),
    "Kupang": (-10.1772, 123.6070, 8.0), "Ambon": (-3.6954, 128.1814, 9.0),
    "Sorong": (-0.8762, 131.2558, 9.0), "Jayapura": (-2.5916, 140.6690, 9.0),
}

# Deteksi lokasi GPS memakai streamlit-js-eval (get_geolocation), yang memakai
# protokol komponen resmi Streamlit (postMessage) — BUKAN navigasi iframe, karena
# iframe komponen Streamlit di-sandbox dan memblokir navigasi window.parent/top,
# sehingga pendekatan JS mentah sebelumnya gagal secara diam-diam.


@st.cache_resource(show_spinner="Memuat/mengunduh ephemeris DE440s (±32 MB)…")
def _warm_ephemeris():
    qc.load_ephemeris()
    return True


def ensure_ephemeris():
    """Muat ephemeris; bila gagal (mis. tak bisa unduh) tampilkan pesan ramah."""
    try:
        _warm_ephemeris()
    except Exception as e:  # noqa: BLE001
        st.error(
            "Gagal memuat data ephemeris (de440s.bsp). Bila server ini offline, "
            "sertakan file `de440s.bsp` di folder aplikasi, atau pastikan ada "
            f"koneksi internet untuk mengunduhnya sekali.\n\nDetail: {e}"
        )
        st.stop()


def user_today(tz_offset: float) -> dt.date:
    """Tanggal 'hari ini' menurut zona waktu pengguna (bukan jam server/UTC)."""
    return (dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=tz_offset)).date()


# ---------------------------------------------------------------------------
# Acuan waktu resmi (BMKG via NTP, dari sisi server) — penjaga akurasi jam
# ---------------------------------------------------------------------------
# Urutan berjenjang: BMKG dulu, lalu cadangan bila terblokir.
NTP_SERVERS = [
    ("ntp.bmkg.go.id", "BMKG"),
    ("time.bmkg.go.id", "BMKG"),
    ("id.pool.ntp.org", "NTP Indonesia"),
]
# Cadangan berbasis HTTP (ramah hosting cloud yang memblokir UDP/NTP).
HTTP_TIME_APIS = [
    ("https://worldtimeapi.org/api/timezone/Etc/UTC", "unixtime", "WorldTimeAPI"),
    ("https://timeapi.io/api/Time/current/zone?timeZone=UTC", None, "TimeAPI"),
]


def _ntp_offset():
    """Offset (detik) via NTP BMKG/nasional. None bila gagal."""
    try:
        import ntplib
    except Exception:  # noqa: BLE001
        return None
    client = ntplib.NTPClient()
    for host, label in NTP_SERVERS:
        try:
            r = client.request(host, version=3, timeout=1.5)
            return float(r.offset), label
        except Exception:  # noqa: BLE001
            continue
    return None


def _http_offset():
    """Offset (detik) via HTTP time API (cadangan cloud). None bila gagal."""
    import json
    import urllib.request
    for url, key, label in HTTP_TIME_APIS:
        try:
            t0 = time.time()
            with urllib.request.urlopen(url, timeout=3) as resp:
                data = json.load(resp)
            rtt = (time.time() - t0) / 2.0
            if key and key in data:                       # unixtime detik
                server_now = float(data[key]) + rtt
            elif "dateTime" in data:                      # ISO string (timeapi.io)
                iso = data["dateTime"].split(".")[0]
                server_now = dt.datetime.fromisoformat(iso).replace(
                    tzinfo=dt.timezone.utc).timestamp() + rtt
            else:
                continue
            return server_now - time.time(), label
        except Exception:  # noqa: BLE001
            continue
    return None


@st.cache_data(ttl=300, show_spinner="Menyelaraskan waktu ke BMKG…")
def get_time_reference() -> dict:
    """
    Offset jam server terhadap waktu resmi, dalam detik (di-cache 5 menit).
    Urutan: NTP BMKG/nasional → HTTP time API (cadangan cloud) → jam perangkat.
    """
    res = _ntp_offset() or _http_offset()
    if res is None:
        return {"ok": False, "offset": 0.0, "source": "jam perangkat"}
    return {"ok": True, "offset": res[0], "source": res[1]}


def authoritative_now_ms(ref: dict) -> int:
    """Epoch milidetik menurut waktu acuan (0 bila tak tersedia)."""
    if not ref.get("ok"):
        return 0
    return int((time.time() + ref["offset"]) * 1000)


# ---------------------------------------------------------------------------
# Auto-lokasi: kota / GPS / tempel desimal — mengisi kolom DMS otomatis
# ---------------------------------------------------------------------------
def _dec_parts(value, is_lat):
    hemi = ("S" if value < 0 else "N") if is_lat else ("W" if value < 0 else "E")
    v = abs(value)
    d = int(v)
    mf = (v - d) * 60.0
    m = int(mf)
    s = round((mf - m) * 60.0, 2)
    if s >= 60.0:
        s -= 60.0
        m += 1
    if m >= 60:
        m -= 60
        d += 1
    return d, m, s, hemi


def _set_coord_state(lat, lon):
    for prefix, val, is_lat in (("lat", lat, True), ("lon", lon, False)):
        d, m, s, h = _dec_parts(val, is_lat)
        st.session_state[f"{prefix}_d"] = d
        st.session_state[f"{prefix}_m"] = m
        st.session_state[f"{prefix}_s"] = float(s)
        st.session_state[f"{prefix}_h"] = h


def _maybe_apply_city(city):
    if city == MANUAL_OPT:
        st.session_state["_last_city"] = city
        return
    if st.session_state.get("_last_city") == city:
        return
    lat, lon, off = CITIES[city]
    _set_coord_state(lat, lon)
    st.session_state["tz_label"] = OFF2LABEL.get(off, "WIB (UTC+7)")
    st.session_state["_last_city"] = city


def _apply_gps_result():
    """Ambil lokasi dari streamlit-js-eval bila kotak deteksi GPS dicentang."""
    if not st.session_state.get("_gps_on"):
        return
    loc = get_geolocation(component_key="gps_loc")
    if not loc:
        st.caption("⏳ Menunggu izin lokasi dari browser…")
        return
    if "error" in loc:
        code = loc["error"].get("code")
        msg = loc["error"].get("message", "")
        if code == 1:
            st.error("Izin lokasi ditolak di browser. Aktifkan lewat ikon 🔒/ⓘ di address bar, lalu coba lagi.")
        else:
            st.warning(f"Gagal mengambil lokasi (kode {code}): {msg}")
        return
    lat = loc["coords"]["latitude"]
    lon = loc["coords"]["longitude"]
    if st.session_state.get("_last_gps") != (lat, lon):
        _set_coord_state(lat, lon)
        st.session_state["_last_gps"] = (lat, lon)
        st.success(f"📍 Lokasi terdeteksi: {lat:.5f}, {lon:.5f}")


def _maybe_apply_paste(text):
    """Terima koordinat desimal seperti '-6.98, 109.61' (format Google Maps)."""
    if not text or st.session_state.get("_last_paste") == text:
        return
    parts = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
    if len(parts) >= 2:
        lat, lon = float(parts[0]), float(parts[1])
        if -90 <= lat <= 90 and -180 <= lon <= 180:
            _set_coord_state(lat, lon)
            st.session_state["_last_paste"] = text
            st.rerun()


# ---------------------------------------------------------------------------
# SIDEBAR — hanya yang esensial; sisanya di "Lanjutan"
# ---------------------------------------------------------------------------
def sidebar_inputs() -> dict:
    st.sidebar.header("⚙️ Pengaturan")

    for k, v in {"lat_d": 6, "lat_m": 59, "lat_s": 0.0, "lat_h": "S",
                 "lon_d": 109, "lon_m": 43, "lon_s": 0.0, "lon_h": "E"}.items():
        st.session_state.setdefault(k, v)

    # --- Lokasi ---
    st.sidebar.subheader("🏙️ Lokasi")
    city = st.sidebar.selectbox("Pilih kota", [MANUAL_OPT] + list(CITIES.keys()), key="city_sel")
    _maybe_apply_city(city)

    paste = st.sidebar.text_input("Atau tempel koordinat (dari Google Maps)",
                                  placeholder="-6.98, 109.61")
    _maybe_apply_paste(paste)

    with st.sidebar.expander("📡 Deteksi lokasi (GPS)"):
        st.checkbox("Aktifkan deteksi GPS", key="_gps_on",
                    help="Browser akan meminta izin akses lokasi.")
        _apply_gps_result()

    with st.sidebar.expander("Koordinat manual (derajat-menit-detik)"):
        st.markdown("**Lintang**")
        la, lb, lc, ld = st.columns(4)
        la.number_input("°", 0, 90, key="lat_d")
        lb.number_input("′", 0, 59, key="lat_m")
        lc.number_input("″", 0.0, 59.999, format="%.2f", key="lat_s")
        ld.radio("Arah", ["N", "S"], key="lat_h")
        st.markdown("**Bujur**")
        oa, ob, oc, od = st.columns(4)
        oa.number_input("°", 0, 180, key="lon_d")
        ob.number_input("′", 0, 59, key="lon_m")
        oc.number_input("″", 0.0, 59.999, format="%.2f", key="lon_s")
        od.radio("Arah", ["E", "W"], key="lon_h")

    lat = qc.dms_to_decimal(st.session_state["lat_d"], st.session_state["lat_m"],
                            st.session_state["lat_s"], st.session_state["lat_h"])
    lon = qc.dms_to_decimal(st.session_state["lon_d"], st.session_state["lon_m"],
                            st.session_state["lon_s"], st.session_state["lon_h"])
    st.sidebar.caption(f"📌 {qc.decimal_to_dms(lat,'lat')}, {qc.decimal_to_dms(lon,'lon')}")

    # --- Zona waktu (otomatis dari kota; bisa diubah) ---
    tz_label = st.sidebar.selectbox("Zona Waktu", list(TZ_OPTIONS.keys()), index=0, key="tz_label")
    if TZ_OPTIONS[tz_label] is None:
        tz_offset = st.sidebar.number_input("Offset UTC kustom (jam)", value=7.0, step=0.5, format="%.1f")
        tz_short = f"UTC{tz_offset:+g}"
    else:
        tz_offset = TZ_OPTIONS[tz_label]
        tz_short = tz_label.split()[0]

    # --- Tanggal (default: hari ini menurut zona waktu pengguna) ---
    the_date = st.sidebar.date_input("📅 Tanggal Pengukuran", value=user_today(tz_offset))

    # --- Tiang ---
    height = st.sidebar.number_input("📏 Tinggi tiang tegak (meter)", min_value=0.05,
                                     value=1.0, step=0.1, format="%.2f")

    # --- Identitas untuk laporan ---
    nama = st.sidebar.text_input("🏷️ Nama lokasi / identitas ", "")

    # --- Lanjutan ---
    with st.sidebar.expander("🔧 Lanjutan"):
        elev = st.number_input("Elevasi (meter)", value=0.0, step=1.0,
                               help="Praktis tidak memengaruhi arah/azimuth Matahari.")
        step = st.radio("Kerapatan sudut busur ΔA (derajat)", [5, 1], index=0, horizontal=True,
                        help="Sudut busur dibulatkan ke kelipatan ini. 5° = ringkas, "
                             "1° = rinci (menangkap lebih banyak peluang).")

    return {
        "lat": lat, "lon": lon, "elev": elev,
        "tz_offset": tz_offset, "tz_short": tz_short,
        "the_date": the_date, "height": height, "step": int(step),
        "nama": nama.strip(),
    }


# ---------------------------------------------------------------------------
# SIMULASI (matplotlib): tampak samping + tampak atas (kompas + busur)
# ---------------------------------------------------------------------------
def _polar_xy(az_deg, r):
    a = math.radians(az_deg)
    return r * math.sin(a), r * math.cos(a)


def render_simulation(a_k, sol, delta_a, session_key, height):
    fig, (axs, axt) = plt.subplots(1, 2, figsize=(11, 5.2))

    # (a) TAMPAK SAMPING
    alt = max(sol.sun_altitude, 0.5)
    L = height / math.tan(math.radians(alt))
    axs.plot([0, 0], [0, height], color="#333", lw=6, solid_capstyle="round")
    axs.plot([0, L], [0, 0], color="#c0392b", lw=4)
    axs.plot([0, L], [height, 0], color="#f39c12", lw=2, ls="--")
    axs.plot([0], [height], "o", color="#333", ms=6)
    axs.add_patch(Arc((L, 0), L * 0.5, L * 0.5, angle=0,
                      theta1=180 - alt, theta2=180, color="#f39c12", lw=1.5))
    axs.annotate(f"alt {alt:.1f}°", (L * 0.72, height * 0.12), color="#b9770e", fontsize=9)
    axs.annotate("tongkat", (0.02 * max(L, 1), height * 0.5), fontsize=9)
    axs.annotate(f"bayangan = {L:.3f} m", (L * 0.15, -height * 0.16), color="#c0392b", fontsize=9)
    axs.plot([0], [0], "o", color="black", ms=4)
    axs.annotate("O", (-0.04 * max(L, 1), -height * 0.14), fontsize=10, weight="bold")
    axs.annotate("B", (L, -height * 0.14), fontsize=10, weight="bold", color="#c0392b")
    axs.set_title("Tampak Samping — Tongkat & Bayangan")
    axs.set_aspect("equal")
    axs.set_xlim(-0.15 * max(L, 1), L * 1.15 + 0.1)
    axs.set_ylim(-height * 0.4, height * 1.25)
    axs.axis("off")

    # (b) TAMPAK ATAS
    R = 1.0
    axt.add_patch(plt.Circle((0, 0), R, fill=False, color="#bbb", lw=1))
    for lbl, az in [("U", 0), ("T", 90), ("S", 180), ("B", 270)]:
        x, y = _polar_xy(az, R * 1.12)
        axt.annotate(lbl, (x, y), ha="center", va="center", fontsize=10, color="#888")
    bx, by = _polar_xy(sol.shadow_azimuth, R)
    axt.annotate("", xy=(bx, by), xytext=(0, 0),
                 arrowprops=dict(arrowstyle="-|>", color="#c0392b", lw=3))
    axt.annotate("Bayangan (O→B)", (bx * 1.05, by * 1.05), color="#c0392b", fontsize=9, ha="center")
    kx, ky = _polar_xy(a_k, R)
    axt.annotate("", xy=(kx, ky), xytext=(0, 0),
                 arrowprops=dict(arrowstyle="-|>", color="#12805a", lw=3))
    axt.annotate("KIBLAT", (kx * 1.08, ky * 1.08), color="#0e5c3f", fontsize=10,
                 ha="center", weight="bold")
    sx, sy = _polar_xy(sol.sun_azimuth, R * 0.75)
    axt.plot([sx], [sy], "o", color="#f1c40f", ms=16, mec="#e67e22")
    axt.annotate("Matahari", (sx, sy - 0.16), ha="center", va="center", fontsize=8, color="#b9770e")
    sweep = delta_a if session_key == "pagi" else -delta_a
    steps = max(int(abs(delta_a)), 2)
    arc_r = R * 0.45
    xs, ys = [], []
    for k in range(steps + 1):
        az = sol.shadow_azimuth + sweep * (k / steps)
        x, y = _polar_xy(az, arc_r)
        xs.append(x); ys.append(y)
    axt.plot(xs, ys, color="#2c3e50", lw=2)
    axt.annotate("", xy=(xs[-1], ys[-1]), xytext=(xs[-2], ys[-2]),
                 arrowprops=dict(arrowstyle="-|>", color="#2c3e50", lw=2))
    arah = "KANAN" if session_key == "pagi" else "KIRI"
    mx, my = _polar_xy(sol.shadow_azimuth + sweep * 0.5, arc_r * 1.25)
    axt.annotate(f"ΔA={delta_a:g}° ke {arah}", (mx, my), ha="center", fontsize=9,
                 color="#2c3e50", weight="bold")
    axt.plot([0], [0], "o", color="black", ms=5)
    axt.set_title("Tampak Atas — Busur dari Bayangan ke Kiblat")
    axt.set_aspect("equal")
    axt.set_xlim(-1.35, 1.35)
    axt.set_ylim(-1.35, 1.35)
    axt.axis("off")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Hitung mundur + aba-aba suara (komponen HTML/JS)
# ---------------------------------------------------------------------------
COUNTDOWN_HTML = """
<div style="font-family:system-ui,Segoe UI,Roboto,sans-serif;text-align:center;
            padding:14px;border-radius:14px;border:1px solid #e5e7eb;background:#fafafa;">
  <div style="font-size:13px;color:#6b7280;">Hitung mundur menuju eksekusi</div>
  <div style="font-size:13px;color:#374151;margin-bottom:4px;">🎯 __LABEL__</div>
  <div id="cd" style="font-size:52px;font-weight:800;line-height:1.1;color:#0e5c3f;">--:--:--</div>
  <div id="status" style="font-size:15px;color:#374151;min-height:20px;margin-top:2px;"></div>
  <button id="arm" style="margin-top:8px;padding:8px 16px;border:0;border-radius:10px;
          background:#12805a;color:#fff;font-size:14px;cursor:pointer;">
    🔔 Aktifkan hitung mundur &amp; suara
  </button>
  <div id="clocknote" style="font-size:11px;color:#9ca3af;margin-top:8px;">
    Menyelaraskan waktu…
  </div>
</div>
<script>
(function(){
  var targetMs = __TARGET_MS__;
  var authNow  = __AUTH_NOW_MS__;   // waktu acuan (BMKG) saat halaman dimuat; 0 = tak ada
  var src      = "__SRC__";
  // Koreksi jam perangkat terhadap waktu acuan (sekali, saat load).
  var clockOffset = (authNow > 0) ? (authNow - Date.now()) : 0;
  function corrNow(){ return Date.now() + clockOffset; }
  var note = document.getElementById('clocknote');
  if(authNow > 0){
    var errS = Math.abs(clockOffset)/1000;
    var arah = clockOffset < 0 ? 'cepat' : 'lambat';
    if(errS < 1){
      note.innerHTML = '🕐 Acuan waktu: <b>'+src+'</b> · jam perangkat akurat (±'+errS.toFixed(1)+' s).';
    } else {
      note.innerHTML = '🕐 Acuan: <b>'+src+'</b> · jam perangkat '+arah+' ~'+errS.toFixed(0)
                       +' s — hitung mundur sudah dikoreksi ke '+src+'.';
      note.style.color = errS > 3 ? '#c0392b' : '#b9770e';
    }
  } else {
    note.innerHTML = '🕐 Acuan: <b>jam perangkat</b> (BMKG tak terjangkau). '
                   + 'Pastikan tanggal &amp; waktu HP mode otomatis.';
    note.style.color = '#b9770e';
  }
  var audio=null, armed=false, lastBeep=null;
  function beep(freq,dur,vol){
    if(!audio) return;
    var o=audio.createOscillator(), g=audio.createGain();
    o.type='sine'; o.frequency.value=freq;
    o.connect(g); g.connect(audio.destination);
    var t=audio.currentTime;
    g.gain.setValueAtTime(0.0001,t);
    g.gain.exponentialRampToValueAtTime(vol,t+0.01);
    g.gain.exponentialRampToValueAtTime(0.0001,t+dur);
    o.start(t); o.stop(t+dur+0.02);
  }
  var arm=document.getElementById('arm');
  arm.onclick=function(){
    audio=new (window.AudioContext||window.webkitAudioContext)();
    audio.resume(); armed=true; beep(880,0.12,0.25);
    arm.textContent='🔔 Suara aktif — siap'; arm.style.background='#0e5c3f';
  };
  function two(n){ return (n<10?'0':'')+n; }
  function fmt(ms){
    var s=Math.floor(ms/1000);
    var d=Math.floor(s/86400); s-=d*86400;
    var h=Math.floor(s/3600); s-=h*3600;
    var m=Math.floor(s/60); s-=m*60;
    if(d>0) return d+' hari '+two(h)+':'+two(m)+':'+two(s);
    if(h>0) return two(h)+':'+two(m)+':'+two(s);
    return two(m)+':'+two(s);
  }
  var cd=document.getElementById('cd'), status=document.getElementById('status');
  function tick(){
    var diff=targetMs-corrNow();
    if(diff>0){
      cd.textContent=fmt(diff);
      var s=Math.ceil(diff/1000);
      if(s<=10){ cd.style.color='#c0392b'; status.textContent='Bersiap menandai bayangan…'; }
      else { cd.style.color='#0e5c3f'; status.textContent=''; }
      if(armed && s<=5 && s>=1 && lastBeep!==s){ beep(880,0.12,0.28); lastBeep=s; }
    } else if(diff>-2000){
      cd.textContent='SEKARANG!'; cd.style.color='#c0392b';
      status.textContent='⏺ Tandai pangkal (O) & ujung bayangan (B) sekarang!';
      if(armed && lastBeep!=='now'){ beep(1320,0.7,0.35); lastBeep='now'; }
    } else {
      cd.textContent='Waktu telah lewat'; cd.style.color='#9ca3af'; status.textContent='';
    }
  }
  setInterval(tick,100); tick();
})();
</script>
"""


def render_countdown(sol, inp, ref):
    epoch_ms = int(sol.local_time.timestamp() * 1000)
    label = f"{sol.time_str()} {inp['tz_short']} · {tgl_id(inp['the_date'])}"
    html = (COUNTDOWN_HTML
            .replace("__TARGET_MS__", str(epoch_ms))
            .replace("__AUTH_NOW_MS__", str(authoritative_now_ms(ref)))
            .replace("__SRC__", ref.get("source", "jam perangkat"))
            .replace("__LABEL__", label))
    components.html(html, height=250)


# ---------------------------------------------------------------------------
# Instruksi & laporan
# ---------------------------------------------------------------------------
def field_instructions(session_key, delta_a, jam, tz_short):
    arah = "KANAN (searah jarum jam)" if session_key == "pagi" else "KIRI (berlawanan jarum jam)"
    st.markdown("#### 🧭 Panduan Eksekusi Lapangan")
    st.markdown(
        f"""
1. **Ratakan papan** dengan waterpass, pasang tiang/benang **tegak lurus (90°)**.
2. Tepat pukul **{jam} {tz_short}**, tandai pangkal tiang **(O)** dan ujung bayangan **(B)**.
3. Tempelkan busur derajat dengan **0° berimpit pada garis O→B** (menghadap ujung bayangan).
4. Dari garis bayangan, **putar {delta_a:g}° ke arah {arah}**.
5. Tarik garis dari O melewati angka tersebut — **inilah Arah Kiblat sejati.**

> Berdirilah di **O menghadap ujung bayangan (B)**; "kanan/kiri" mengacu pada sudut pandang ini.
"""
    )


def build_report(inp, a_k, sol):
    session_key = sol.session
    delta_a = sol.delta_a
    a_s = sol.sun_azimuth  # waktu tetap -> A_s = azimuth Matahari saat itu
    arah = "KANAN (searah jarum jam)" if session_key == "pagi" else "KIRI (berlawanan jarum jam)"
    panjang = f"{sol.shadow_length:.3f} m" if sol.shadow_length is not None else "tak terdefinisi"
    sesi = "Pagi" if session_key == "pagi" else "Sore"
    return f"""LAPORAN HASIL PENENTUAN ARAH KIBLAT
Metode: Selisih Azimuth Matahari Harian (Rashdul Qiblah Harian)
========================================================

DATA LOKASI
- Lintang  : {qc.decimal_to_dms(inp['lat'], 'lat')}  ({inp['lat']:.6f}°)
- Bujur    : {qc.decimal_to_dms(inp['lon'], 'lon')}  ({inp['lon']:.6f}°)
- Elevasi  : {inp['elev']:.1f} m
- Zona     : {inp['tz_short']}
- Tanggal  : {tgl_id(inp['the_date'])}
- Sesi     : {sesi}

HASIL PERHITUNGAN
- Azimuth Kiblat (A_k)          : {qc.decimal_to_dms(a_k, 'az')}   ({a_k:.4f}°)
- Target Azimuth Matahari (A_s) : {qc.decimal_to_dms(a_s, 'az')}   ({a_s:.4f}°)
- Target Selisih Azimuth (ΔA)   : {delta_a:g}°
- WAKTU EKSEKUSI                 : {sol.time_str()} {inp['tz_short']}
- Azimuth Matahari saat itu     : {qc.decimal_to_dms(sol.sun_azimuth, 'az')}
- Altitude Matahari saat itu    : {qc.decimal_to_dms(sol.sun_altitude, 'alt')}
- Tinggi tiang                  : {inp['height']:.2f} m
- Panjang bayangan prediksi     : {panjang}
- Azimuth bayangan              : {qc.decimal_to_dms(sol.shadow_azimuth, 'az')}

LANGKAH LAPANGAN
1. Ratakan papan (waterpass), pasang tiang tegak lurus 90°.
2. Pukul {sol.time_str()} {inp['tz_short']}: tandai pangkal (O) & ujung bayangan (B).
3. Tempelkan busur 0° pada garis O->B.
4. Putar {delta_a:g}° ke arah {arah}.
5. Tarik garis dari O melewati angka tsb = Arah Kiblat sejati.
"""


def _lat(s: str) -> str:
    """Bersihkan teks agar aman untuk font inti PDF (Latin-1)."""
    repl = {"Δ": "d", "→": "->", "≈": "~", "°": chr(176), "–": "-", "—": "-",
            "’": "'", "“": '"', "”": '"', "×": "x", "±": "+/-"}
    for a, b in repl.items():
        s = s.replace(a, b)
    return s.encode("latin-1", "replace").decode("latin-1")


def build_pdf(inp, a_k, sol) -> bytes:
    """Laporan PDF polos berisi data, hasil, gambar simulasi, langkah, sumber."""
    from fpdf import FPDF
    from fpdf.enums import XPos, YPos

    session_key, delta_a, a_s = sol.session, sol.delta_a, sol.sun_azimuth
    sesi = "Pagi" if session_key == "pagi" else "Sore"
    arah = "KANAN (searah jarum jam)" if session_key == "pagi" else "KIRI (berlawanan jarum jam)"
    panjang = f"{sol.shadow_length:.3f} m" if sol.shadow_length is not None else "tak terdefinisi"
    nama = inp.get("nama") or "-"

    # Gambar simulasi -> PNG sementara
    fig = render_simulation(a_k, sol, delta_a, session_key, inp["height"])
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    fig.savefig(tmp.name, dpi=110, bbox_inches="tight")
    plt.close(fig)

    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(True, 15)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _lat("LAPORAN PENENTUAN ARAH KIBLAT"), new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, _lat("Metode Selisih Azimuth Matahari Harian (Rashdul Qiblah Harian)"),
             new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.ln(3)

    def row(label, value):
        pdf.set_x(pdf.l_margin)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(55, 6, _lat(label), border=0)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 6, _lat(str(value)), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, _lat("Nama lokasi / Identitas: ") + _lat(nama), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(1)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "DATA LOKASI", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    row("Lintang", f"{qc.decimal_to_dms(inp['lat'],'lat')}  ({inp['lat']:.6f})")
    row("Bujur", f"{qc.decimal_to_dms(inp['lon'],'lon')}  ({inp['lon']:.6f})")
    row("Elevasi", f"{inp['elev']:.1f} m")
    row("Zona waktu", inp["tz_short"])
    row("Tanggal", tgl_id(inp["the_date"]))
    row("Sesi", sesi)
    pdf.ln(1)

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "HASIL PERHITUNGAN", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    row("Azimuth Kiblat (A_k)", qc.decimal_to_dms(a_k, "az"))
    row("Target Azimuth Matahari", qc.decimal_to_dms(a_s, "az"))
    row("Selisih Azimuth (dA)", f"{delta_a:g} deg")
    row("Waktu eksekusi", f"{sol.time_str()} {inp['tz_short']}")
    row("Altitude Matahari", qc.decimal_to_dms(sol.sun_altitude, "alt"))
    row("Tinggi tiang", f"{inp['height']:.2f} m")
    row("Panjang bayangan", panjang)
    row("Azimuth bayangan", qc.decimal_to_dms(sol.shadow_azimuth, "az"))
    pdf.ln(2)

    y0 = pdf.get_y()
    pdf.image(tmp.name, x=pdf.l_margin, y=y0, w=180)
    pdf.set_y(y0 + 180 * 0.50 + 4)      # tinggi gambar ~ rasio aspek + jeda

    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "LANGKAH LAPANGAN", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    steps = [
        "Ratakan papan (waterpass), pasang tiang tegak lurus 90 derajat.",
        f"Pukul {sol.time_str()} {inp['tz_short']}: tandai pangkal (O) & ujung bayangan (B).",
        "Tempelkan busur 0 derajat pada garis O->B.",
        f"Putar {delta_a:g} derajat ke arah {arah}.",
        "Tarik garis dari O melewati angka tsb = Arah Kiblat sejati.",
    ]
    pdf.set_font("Helvetica", "", 10)
    for i, s in enumerate(steps, 1):
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, 6, _lat(f"{i}. {s}"))
    pdf.ln(2)

    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 5, _lat(
        "Sumber: efemeris NASA/JPL DE440s (Skyfield); koordinat Kakbah acuan Kemenag RI. "
        "Akurasi lapangan dibatasi ketegakan tiang, kerataan papan, ketajaman ujung "
        "bayangan, dan pembacaan busur (realistis beberapa menit busur). "
        f"Dicetak: {tgl_id(dt.datetime.now())} {dt.datetime.now().strftime('%H:%M')}."))
    pdf.ln(6)
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", "", 10)
    # pdf.ln(14)
    # pdf.set_x(pdf.l_margin)
    # pdf.cell(90, 6, "(............................)", border=0)
    # pdf.cell(0, 6, "(............................)", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    return bytes(pdf.output())


# ---------------------------------------------------------------------------
# Tampilkan satu waktu terpilih (hero + countdown + simulasi + instruksi)
# ---------------------------------------------------------------------------
def show_solution(inp, a_k, sol, ref):
    session_key = sol.session
    delta_a = sol.delta_a
    a_s = sol.sun_azimuth  # waktu tetap -> A_s = azimuth Matahari saat itu
    sesi = "Pagi" if session_key == "pagi" else "Sore"

    st.markdown(
        f"""
<div style="text-align:center;padding:18px;border-radius:16px;
            background:linear-gradient(135deg,#0e5c3f,#12805a);color:#fff;margin-bottom:10px;">
  <div style="font-size:15px;opacity:.85;letter-spacing:1px;">WAKTU EKSEKUSI</div>
  <div style="font-size:56px;font-weight:800;line-height:1.1;">{sol.time_str()}</div>
  <div style="font-size:17px;opacity:.9;">{inp['tz_short']} · {tgl_id(inp['the_date'])} · Sesi {sesi} · ΔA {delta_a:g}°</div>
</div>
""",
        unsafe_allow_html=True,
    )

    if inp["the_date"] == user_today(inp["tz_offset"]):
        render_countdown(sol, inp, ref)
    else:
        st.caption("⏱️ Hitung mundur aktif hanya bila tanggal pengukuran = hari ini.")

    d1, d2, d3 = st.columns(3)
    d1.metric("Altitude Matahari", qc.decimal_to_dms(sol.sun_altitude, "alt"))
    d2.metric("Panjang Bayangan",
              f"{sol.shadow_length:.3f} m" if sol.shadow_length is not None else "—")
    d3.metric("Target Azimuth Matahari (A_s)", qc.decimal_to_dms(a_s, "az"))

    if sol.sun_altitude < 10:
        st.warning("Altitude rendah: bayangan panjang & ujung kabur.")
    elif sol.sun_altitude > 65:
        st.warning("Altitude tinggi: bayangan pendek.")
    else:
        st.success("Altitude ideal untuk pengukuran bayangan (±15°–60°).")

    st.pyplot(render_simulation(a_k, sol, delta_a, session_key, inp["height"]))
    field_instructions(session_key, delta_a, sol.time_str(), inp["tz_short"])

    report = build_report(inp, a_k, sol)
    with st.expander("🧾 Laporan (untuk lampiran tugas)"):
        cpdf, ctxt = st.columns(2)
        try:
            pdf_bytes = build_pdf(inp, a_k, sol)
            cpdf.download_button(
                "⬇️ Unduh Laporan PDF", data=pdf_bytes,
                file_name=f"laporan_kiblat_{inp['the_date']}_{session_key}_{delta_a:g}.pdf",
                mime="application/pdf", width="stretch")
        except Exception as e:  # noqa: BLE001
            cpdf.caption(f"PDF tak tersedia: {e}")
        ctxt.download_button(
            "⬇️ Unduh Teks (.txt)", data=report,
            file_name=f"laporan_kiblat_{inp['the_date']}_{session_key}_{delta_a:g}.txt",
            mime="text/plain", width="stretch")
        st.code(report, language="text")


# ---------------------------------------------------------------------------
# Tabel semua peluang + penanda "menuju sekarang"
# ---------------------------------------------------------------------------
def _seconds_to_now(sol):
    now = dt.datetime.now(dt.timezone.utc)
    return (sol.local_time.astimezone(dt.timezone.utc) - now).total_seconds()


def nearest_index(rows, the_date, today):
    """Indeks waktu terdekat: bila hari ini, yang terdekat ke depan; jika tak ada, terakhir."""
    if the_date == today:
        future = [(s, i) for i, s in enumerate(rows) if _seconds_to_now(s) >= 0]
        if future:
            return min(future, key=lambda t: _seconds_to_now(t[0]))[1]
    return 0


def build_all_df(rows, the_date, today):
    is_today = the_date == today
    data = []
    for i, s in enumerate(rows):
        sesi = "Pagi" if s.session == "pagi" else "Sore"
        if is_today:
            d = _seconds_to_now(s)
            if d >= 3600:
                sisa = f"{int(d//3600)} j {int((d % 3600)//60)} m lagi"
            elif d >= 0:
                sisa = f"{int(d//60)} m lagi"
            else:
                sisa = "sudah lewat"
        else:
            sisa = "—"
        data.append({
            "No": i + 1, "Waktu": s.time_str(), "Sesi": sesi, "ΔA (°)": s.delta_a,
            "Altitude": qc.decimal_to_dms(s.sun_altitude, "alt"),
            "Bayangan": f"{s.shadow_length:.3f} m" if s.shadow_length is not None else "—",
            "Menuju kini": sisa,
        })
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Panel Rashdul Qiblah harian (ΔA = 0, bayangan langsung = garis Kiblat)
# ---------------------------------------------------------------------------
def render_rashdul(inp, rash, ref):
    st.subheader("🌞 Rashdul Qiblah Harian — bayangan langsung menunjuk Kiblat (tanpa busur)")
    events = rash["events"]
    if not events:
        st.caption("Pada tanggal & lokasi ini, tidak ada momen Matahari tepat searah maupun "
                   "berlawanan Kiblat saat di atas ufuk. Gunakan metode busur ΔA di bawah.")
        return
    for e in events:
        s = e["solution"]
        if e["kind"] == "searah":
            judul = "☀️ Matahari SEARAH Kiblat"
            narasi = ("Matahari tepat di arah Kiblat. Bayangan tongkat menunjuk **berlawanan** "
                      "Kiblat — arah Kiblat = kebalikan bayangan (dari ujung bayangan **B** ke "
                      "pangkal **O**, lurus ke Matahari).")
        else:
            judul = "🌗 Matahari membelakangi Kiblat (sebaliknya 180°)"
            narasi = ("Bayangan tongkat menunjuk **tepat ke Kiblat** — arah Kiblat = garis "
                      "bayangan **O→B** (cukup perpanjang bayangannya).")
        with st.container(border=True):
            st.markdown(f"**{judul}** · pukul **{s.time_str()} {inp['tz_short']}**")
            c1, c2, c3 = st.columns(3)
            c1.metric("Altitude Matahari", qc.decimal_to_dms(s.sun_altitude, "alt"))
            c2.metric("Azimuth Matahari", qc.decimal_to_dms(s.sun_azimuth, "az"))
            c3.metric("Panjang Bayangan",
                      f"{s.shadow_length:.3f} m" if s.shadow_length is not None else "—")
            st.markdown(narasi)
            if s.sun_altitude < 5:
                st.caption("⚠️ Matahari sangat rendah — bayangan panjang & ujungnya kabur.")
            if inp["the_date"] == user_today(inp["tz_offset"]):
                render_countdown(s, inp, ref)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def _sig(inp):
    return (round(inp["lat"], 6), round(inp["lon"], 6), str(inp["the_date"]),
            inp["tz_offset"], inp["height"], inp["elev"], inp["step"])


def main():
    st.title("🕋 BayangKiblat")
    st.caption("Penentu Arah Kiblat berbasis Matahari — Rashdul Qiblah Harian & "
               "Metode Selisih Azimuth · Skyfield + JPL DE440s · Koordinat Kakbah acuan Kemenag RI")

    inp = sidebar_inputs()

    # Penjaga akurasi: kecocokan zona waktu dengan bujur
    expected = inp["lon"] / 15.0
    if abs(inp["tz_offset"] - expected) > 1.5:
        st.warning(f"⚠️ Zona waktu (UTC{inp['tz_offset']:+g}) tampak tak cocok dengan bujur "
                   f"{inp['lon']:.2f}° (perkiraan UTC{expected:+.1f}). Pastikan zona waktu benar — "
                   "salah zona menggeser jam eksekusi.")

    st.caption("🕐 Waktu eksekusi mengacu ke jam resmi **BMKG** (diselaraskan otomatis saat "
               "online). Jika offline, gunakan jam perangkat mode otomatis.")

    hitung = st.button(f"🔮 Hitung Waktu Kiblat — {tgl_id(inp['the_date'])}",
                       type="primary", width="stretch")

    if hitung:
        ensure_ephemeris()
        try:
            res = qc.compute_all(
                lat=inp["lat"], lon=inp["lon"], elev=inp["elev"],
                the_date=inp["the_date"], tz_offset=inp["tz_offset"],
                height=inp["height"], step=inp["step"],
            )
            rash = qc.rashdul_qiblah(
                lat=inp["lat"], lon=inp["lon"], elev=inp["elev"],
                the_date=inp["the_date"], tz_offset=inp["tz_offset"], height=inp["height"],
            )
            st.session_state["snap"] = {"inp": inp, "res": res, "rash": rash, "sig": _sig(inp)}
        except Exception as e:  # noqa: BLE001
            st.error(f"Kesalahan perhitungan: {e}")
            st.stop()

    snap = st.session_state.get("snap")
    if snap is None:
        st.info("Atur **Lokasi** & **Tanggal** di panel kiri, lalu klik tombol **Hitung** di atas.")
        st.stop()

    if snap["sig"] != _sig(inp):
        st.warning("Parameter berubah sejak perhitungan terakhir — klik **Hitung** lagi "
                   "untuk memperbarui hasil.")

    inp = snap["inp"]
    res = snap["res"]
    rash = snap.get("rash")
    a_k = res["a_k"]
    rows = res["rows"]
    today = user_today(inp["tz_offset"])
    ref = get_time_reference()

    st.metric("Azimuth Kiblat lokasi ini (A_k)", qc.decimal_to_dms(a_k, "az"))
    st.divider()

    # --- Rashdul Qiblah harian (bayangan langsung menunjuk Kiblat, tanpa busur) ---
    if rash is not None:
        render_rashdul(inp, rash, ref)
        st.divider()

    # --- Semua waktu peluang metode selisih azimuth (pakai busur ΔA) ---
    if not rows:
        st.warning("Tidak ada waktu ΔA (metode busur) yang layak pada tanggal ini. "
                   "Bila ada, gunakan Rashdul Qiblah harian di atas, atau coba tanggal lain.")
    else:
        st.subheader(f"🗓️ Semua Waktu Peluang — Metode Busur ΔA ({len(rows)} waktu)")
        st.caption("Pagi & sore digabung, urut waktu. Pilih salah satu untuk melihat simulasi "
                   "& hitung mundur. Default: waktu terdekat.")
        st.dataframe(build_all_df(rows, inp["the_date"], today),
                     width="stretch", hide_index=True)

        idx0 = nearest_index(rows, inp["the_date"], today)
        labels = [f"{s.time_str()} · {'Pagi' if s.session=='pagi' else 'Sore'} · ΔA {s.delta_a:g}° "
                  f"· alt {s.sun_altitude:.0f}°" for s in rows]
        choice = st.selectbox("Pilih waktu eksekusi", range(len(rows)),
                              index=idx0, format_func=lambda i: labels[i])
        st.divider()
        show_solution(inp, a_k, rows[choice], ref)

    with st.expander("ℹ️ Catatan akurasi"):
        st.markdown("Waktu berpresisi detik, namun akurasi lapangan dibatasi ketegakan tiang, "
                    "kerataan papan, ketajaman ujung bayangan, dan pembacaan busur — realistis "
                    "beberapa menit busur. ΔA dibulatkan ke derajat bulat; waktu tetap dihitung "
                    "tepat untuk target itu, jadi tidak mengurangi akurasi.")


if __name__ == "__main__":
    main()
