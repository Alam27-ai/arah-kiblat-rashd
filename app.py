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

import csv
import datetime as dt
import io
import math
import os
import tempfile
import time

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates
from streamlit_js_eval import get_geolocation

import papan_charuco as pc
import qibla_core as qc

LOG_UKUR_CV = "log_ukur_cv.csv"
LOG_UKUR_CV_HEADER = [
    "waktu_dicatat", "nama_file", "lat", "lon", "waktu_potret", "tinggi_gnomon_m",
    "kotak_mm", "panjang_bayangan_mm", "delta_a_target_deg", "arah",
    "altitude_efemeris_deg", "altitude_terukur_deg", "selisih_altitude_deg",
    "validasi_altitude_ok", "vonis_mutu_foto", "bias_mutu_deg",
    "sisi_potong_tepi", "posisi_potong_cm",
    "selisih_referensi_deg", "catatan",
]

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

# Rentang ΔA per sesi (dipakai internal; pengguna tak perlu memilih).
RANGE_PAGI = (10, 80)
RANGE_SORE = (100, 170)

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


# ---------------------------------------------------------------------------
# SIDEBAR — hanya yang esensial; sisanya di "Lanjutan"
# ---------------------------------------------------------------------------
def sidebar_inputs(mode: str) -> dict:
    st.sidebar.header("⚙️ Pengaturan")

    _terapkan_exif_pending_lokasi()  # sebelum widget lat/lon/tanggal dibuat

    for k, v in {"lat_d": 6, "lat_m": 59, "lat_s": 0.0, "lat_h": "S",
                 "lon_d": 109, "lon_m": 43, "lon_s": 0.0, "lon_h": "E"}.items():
        st.session_state.setdefault(k, v)

    # --- Lokasi ---
    st.sidebar.subheader("🏙️ Lokasi")
    st.sidebar.checkbox("📡 Deteksi lokasi otomatis (GPS)", key="_gps_on",
                        help="Browser akan meminta izin akses lokasi.")
    _apply_gps_result()

    with st.sidebar.expander("Koordinat manual (derajat-menit-detik)", expanded=True):
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
    # key="the_date" sengaja ditambahkan (bukan cuma dibiarkan auto) supaya
    # bisa diisi otomatis dari EXIF foto di mode "Ukur dari Foto" — lihat
    # _isi_otomatis_dari_exif().
    st.session_state.setdefault("the_date", user_today(tz_offset))
    the_date = st.sidebar.date_input("📅 Tanggal Pengukuran", key="the_date")

    # --- Tiang ---
    # Cuma relevan untuk mode Terjadwal & Bebas waktu (tiang/tongkat FISIK
    # sungguhan di lapangan). Mode "Ukur dari Foto" punya gnomon PAPAN sendiri
    # (field "Tinggi gnomon di papan" di dalam mode itu, skalanya beda jauh —
    # cm bukan meter) yang benar-benar dipakai untuk hitungan §7.3 di sana;
    # field ini disembunyikan di mode itu supaya tidak ada dua "tinggi
    # gnomon/tiang" nampang bersamaan dan membingungkan mana yang dipakai.
    if not mode.startswith("📷"):
        height = st.sidebar.number_input("📏 Tinggi tiang tegak (meter)", min_value=0.05,
                                         value=1.0, step=0.1, format="%.2f", key="height_tiang")
    else:
        st.session_state.setdefault("height_tiang", 1.0)
        height = st.session_state["height_tiang"]

    # --- Identitas untuk laporan --- (cuma dipakai laporan PDF mode Terjadwal)
    if mode.startswith("🗓️"):
        nama = st.sidebar.text_input("🏷️ Nama lokasi / identitas ", key="nama_lokasi")
    else:
        st.session_state.setdefault("nama_lokasi", "")
        nama = st.session_state["nama_lokasi"]

    # --- Lanjutan ---
    with st.sidebar.expander("🔧 Lanjutan"):
        elev = st.number_input("Elevasi (meter)", value=0.0, step=1.0,
                               help="Praktis tidak memengaruhi arah/azimuth Matahari.")
        # "step" cuma dipakai pencarian busur ΔA bulat mode Terjadwal.
        if mode.startswith("🗓️"):
            step = st.radio("Kerapatan sudut busur ΔA (derajat)", [5, 1], index=0, horizontal=True,
                            help="Sudut busur dibulatkan ke kelipatan ini. 5° = ringkas, "
                                 "1° = rinci (menangkap lebih banyak peluang).")
        else:
            step = 5

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
# MODE BEBAS WAKTU (ΔA presisi, tidak dibulatkan — untuk pengukuran dibantu CV)
# ---------------------------------------------------------------------------
def render_instant_mode(inp, ref):
    st.caption(
        "Sudut di sini **tidak dibulatkan** — cocok untuk siapa pun yang membaca "
        "sudutnya lewat citra (bukan busur derajat manual). Pilih waktu apa saja "
        "(termasuk sekarang), lalu putar sesuai sudut hasil hitung."
    )

    tz = dt.timezone(dt.timedelta(hours=inp["tz_offset"]))
    is_today = inp["the_date"] == user_today(inp["tz_offset"])

    if "instant_hm" not in st.session_state:
        now_t = dt.datetime.now(tz).time() if is_today else dt.time(12, 0, 0)
        st.session_state["instant_hm"] = now_t.replace(second=0, microsecond=0)
        st.session_state["instant_sec"] = now_t.second if is_today else 0

    def _set_now():
        # Callback dijalankan SEBELUM widget dibuat ulang, sehingga aman
        # mengubah session_state kunci widget di sini (tidak boleh dari luar callback).
        now_t2 = dt.datetime.now(tz).time()
        st.session_state["instant_hm"] = now_t2.replace(second=0, microsecond=0)
        st.session_state["instant_sec"] = now_t2.second

    c1, c2, c3 = st.columns([2, 1, 1])
    c1.time_input("⏱️ Waktu pengukuran", key="instant_hm", step=60)
    c2.number_input("Detik", 0, 59, key="instant_sec")
    c3.button("🔄 Sekarang", width="stretch", help="Isi waktu saat ini", on_click=_set_now)

    t = st.session_state["instant_hm"].replace(second=int(st.session_state["instant_sec"]))
    dt_local = dt.datetime.combine(inp["the_date"], t, tzinfo=tz)

    ensure_ephemeris()
    try:
        out = qc.solve_instant(inp["lat"], inp["lon"], inp["elev"], dt_local, inp["height"])
    except Exception as e:  # noqa: BLE001
        st.error(f"Kesalahan perhitungan: {e}")
        return

    a_k, sol, arah = out["a_k"], out["solution"], out["arah"]

    if sol.sun_altitude <= 0:
        st.warning("☾ Matahari di bawah ufuk pada waktu ini — tidak ada bayangan. "
                   "Pilih waktu lain (siang hari).")
        return

    st.markdown(
        f"""
<div style="text-align:center;padding:18px;border-radius:16px;
            background:linear-gradient(135deg,#0e5c3f,#12805a);color:#fff;margin-bottom:10px;">
  <div style="font-size:15px;opacity:.85;letter-spacing:1px;">SUDUT PUTAR DARI BAYANGAN</div>
  <div style="font-size:56px;font-weight:800;line-height:1.1;">{sol.delta_a:.3f}°</div>
  <div style="font-size:17px;opacity:.9;">ke arah {arah.upper()} · {t.strftime('%H:%M:%S')} {inp['tz_short']} · {tgl_id(inp['the_date'])}</div>
</div>
""",
        unsafe_allow_html=True,
    )

    now_local = dt.datetime.now(tz)
    if dt_local >= now_local - dt.timedelta(seconds=2):
        render_countdown(sol, inp, ref)
    elif is_today:
        lewat_menit = int((now_local - dt_local).total_seconds() // 60)
        st.caption(f"🕓 Waktu ini sudah lewat ({lewat_menit} menit lalu) — cocok untuk "
                   "pratinjau/uji perhitungan, bukan eksekusi langsung.")

    d1, d2, d3 = st.columns(3)
    d1.metric("Altitude Matahari", qc.decimal_to_dms(sol.sun_altitude, "alt"))
    d2.metric("Panjang Bayangan",
              f"{sol.shadow_length:.3f} m" if sol.shadow_length is not None else "—")
    d3.metric("Azimuth Kiblat (A_k)", qc.decimal_to_dms(a_k, "az"))

    if sol.sun_altitude < 10:
        st.warning("Altitude rendah: bayangan panjang & ujung kabur.")
    elif sol.sun_altitude > 65:
        st.warning("Altitude tinggi: bayangan pendek — juga memperbesar dampak "
                   "kemiringan gnomon (galat ≈ kemiringan × tan(altitude)).")
    else:
        st.success("Altitude ideal untuk pengukuran bayangan (±15°–60°).")

    # session_key "pagi"~kanan / "sore"~kiri dipakai ulang murni sebagai kode arah,
    # bukan penanda sesi pagi/sore sungguhan — konvensi putarnya sama persis.
    session_key = "pagi" if arah == "kanan" else "sore"
    st.pyplot(render_simulation(a_k, sol, sol.delta_a, session_key, inp["height"]))
    field_instructions(session_key, sol.delta_a, t.strftime("%H:%M:%S"), inp["tz_short"])

    st.caption(
        "📸 Ambil foto papan kalibrasi + bayangan pada detik ini, lalu ukur sudutnya "
        "lewat pipeline CV (lihat `RENCANA_CV.md`). Karena sudutnya presisi (bukan "
        "kelipatan bulat), pembacaan manual pakai busur derajat kurang cocok di mode ini."
    )


# ---------------------------------------------------------------------------
# MENU BARU: "Ukur dari Foto" — pengukuran sudut Kiblat dari papan ChArUco
# ---------------------------------------------------------------------------
def baca_exif_foto(raw_bytes: bytes) -> dict:
    """Baca metadata GPS + waktu potret dari EXIF foto, kalau ada.

    Mengembalikan dict {lat, lon, tanggal (date), jam (time)} — key yang
    infonya tidak ada di EXIF cukup dihilangkan (bukan error). Ini NORMAL,
    bukan tanda foto rusak: kompresi WhatsApp/medsos menghapus semua EXIF,
    dan banyak kamera HP tidak menyematkan GPS kalau izin lokasi untuk
    aplikasi kameranya tidak aktif saat memotret. Fungsi ini dibuat untuk
    mempersingkat input di mode "Ukur dari Foto" KALAU datanya ada — bukan
    prasyarat, alur manual/GPS-browser yang sudah ada tetap jadi andalan.
    """
    hasil: dict = {}
    try:
        img = Image.open(io.BytesIO(raw_bytes))
        exif = img.getexif()
        if not exif:
            return hasil
    except Exception:
        return hasil

    try:
        exif_ifd = exif.get_ifd(0x8769)  # Exif IFD
        dt_str = exif_ifd.get(0x9003) or exif.get(0x0132)  # DateTimeOriginal / DateTime
        if dt_str:
            dt_obj = dt.datetime.strptime(dt_str, "%Y:%m:%d %H:%M:%S")
            hasil["tanggal"] = dt_obj.date()
            hasil["jam"] = dt_obj.time()
    except Exception:
        pass

    try:
        gps_ifd = exif.get_ifd(0x8825)  # GPS IFD
        if gps_ifd:
            lat_dms, lat_ref = gps_ifd.get(2), gps_ifd.get(1)
            lon_dms, lon_ref = gps_ifd.get(4), gps_ifd.get(3)
            if lat_dms and lon_dms:
                def _dms_ke_desimal(dms):
                    d, m, s = float(dms[0]), float(dms[1]), float(dms[2])
                    return d + m / 60 + s / 3600
                lat = _dms_ke_desimal(lat_dms)
                lon = _dms_ke_desimal(lon_dms)
                if lat_ref in ("S", b"S"):
                    lat = -lat
                if lon_ref in ("W", b"W"):
                    lon = -lon
                hasil["lat"] = lat
                hasil["lon"] = lon
    except Exception:
        pass

    return hasil


def _isi_otomatis_dari_exif(raw_bytes: bytes):
    """Coba persingkat input mode "Ukur dari Foto": isi lat/lon/tanggal/jam
    otomatis dari EXIF foto yang baru diunggah. Dipanggil SEKALI tiap ada
    foto baru (lihat pemanggilnya) — tidak menimpa perubahan manual pengguna
    di rerun-rerun berikutnya. Kalau EXIF tidak punya info itu (umum), tidak
    melakukan apa-apa selain kasih tahu, dan alur input manual/GPS-browser
    yang sudah ada tetap berfungsi seperti biasa."""
    info = baca_exif_foto(raw_bytes)
    if not info:
        st.session_state["ukur_exif_pesan"] = (
            "info",
            "Metadata (lokasi/waktu) tidak ditemukan di foto ini — umum "
            "terjadi kalau file sudah lewat kompresi WhatsApp/medsos, atau "
            "izin lokasi kamera tidak aktif saat memotret. Isi manual di "
            "atas, atau centang '📡 Deteksi lokasi otomatis (GPS)' di "
            "sidebar (pakai lokasi browser SEKARANG, bukan lokasi saat foto "
            "diambil — hanya cocok kalau Anda mengunggah dari lokasi yang "
            "sama)."
        )
        return

    berubah = []
    if "lat" in info and "lon" in info:
        berubah.append(f"lokasi ({info['lat']:.5f}, {info['lon']:.5f})")
    if "tanggal" in info:
        berubah.append(f"tanggal ({info['tanggal'].strftime('%d-%m-%Y')})")
    if "jam" in info:
        berubah.append(f"jam potret ({info['jam'].strftime('%H:%M:%S')})")
    if not berubah:
        return

    # PENTING: tidak boleh langsung menulis ke session_state milik widget yang
    # SUDAH dibuat di run ini (lat_d, the_date, ukur_hm semua sudah
    # terinstansiasi lebih dulu sebelum titik ini tercapai — Streamlit
    # melarang menulis ke session_state widget setelah ia dibuat pada run
    # yang sama). Simpan dulu sebagai "pending"; baru diterapkan di AWAL
    # sidebar_inputs() / seksi waktu potret pada run berikutnya, sebelum
    # widget-widget itu dibuat lagi — lihat _terapkan_exif_pending_lokasi()
    # dan _terapkan_exif_pending_jam().
    st.session_state["ukur_exif_pending"] = info
    st.session_state["ukur_exif_pesan"] = (
        "success",
        "📸 Terbaca dari metadata foto, otomatis diisikan: " +
        ", ".join(berubah) + ". Periksa dulu di atas sebelum lanjut — "
        "ubah manual kalau ada yang keliru (mis. GPS HP memang bisa "
        "meleset beberapa meter, tidak masalah untuk arah Kiblat; tapi "
        "jam potret sebaiknya tepat)."
    )
    st.rerun()


def _terapkan_exif_pending_lokasi():
    """Terapkan bagian lokasi/tanggal dari EXIF pending. Dipanggil di AWAL
    sidebar_inputs(), SEBELUM widget lat/lon/tanggal dibuat — TIDAK
    membersihkan pending (bagian jam masih perlu dipakai belakangan oleh
    _terapkan_exif_pending_jam, yang jadi pemakai terakhir dan membersihkan)."""
    info = st.session_state.get("ukur_exif_pending")
    if not info:
        return
    if "lat" in info and "lon" in info:
        _set_coord_state(info["lat"], info["lon"])
    if "tanggal" in info:
        st.session_state["the_date"] = info["tanggal"]


def _terapkan_exif_pending_jam():
    """Terapkan bagian jam dari EXIF pending. Dipanggil di awal seksi 'waktu
    potret' render_ukur_foto(), SEBELUM widget ukur_hm/ukur_sec dibuat. Ini
    pemakai terakhir dari 'ukur_exif_pending' — sekalian dibersihkan di sini."""
    info = st.session_state.pop("ukur_exif_pending", None)
    if not info or "jam" not in info:
        return
    st.session_state["ukur_hm"] = info["jam"].replace(second=0, microsecond=0)
    st.session_state["ukur_sec"] = info["jam"].second


def _catat_log_ukur(row: dict):
    """Tambah satu baris ke log_ukur_cv.csv (dibuat kalau belum ada).

    Sengaja dipisah dari log_foto_bayangan.csv yang sudah ada di repo — file
    itu skemanya untuk pencatatan MANUAL (baca_busur_manual_deg, kondisi_langit
    isian tangan), makna kolomnya beda dari hasil otomatis CV di sini. Lebih
    jujur punya berkas sendiri daripada memaksakan ke skema yang tidak cocok.
    """
    baru = not os.path.exists(LOG_UKUR_CV)
    with open(LOG_UKUR_CV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LOG_UKUR_CV_HEADER)
        if baru:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in LOG_UKUR_CV_HEADER})


def _reset_klik():
    for k in ("ukur_titik_B", "ukur_titik_ref"):
        st.session_state.pop(k, None)


def render_ukur_foto(inp, ref):
    st.caption(
        "Foto papan kalibrasi ChArUco (v2, dengan titik gnomon O tetap tercetak "
        "+ skala tepi) beserta bayangannya, lalu tandai **satu titik**: ujung "
        "bayangan. Titik pangkal (O) tidak perlu ditandai — posisinya sudah "
        "diketahui dari cetakan. Aplikasi menghitung arah Kiblat lewat homografi "
        "papan — tidak perlu Utara sejati, tidak perlu busur derajat fisik — dan "
        "menunjukkan di tepi mana serta angka berapa untuk menandai garis Kiblat "
        "secara fisik."
    )

    # --- 1. Unggah foto DULU (foto asli kamera biasanya bawa metadata GPS +
    # jam potret di EXIF-nya) — supaya kalau metadata itu ada, langkah waktu
    # & lokasi di bawah otomatis terisi dan tidak perlu diisi manual dulu
    # sebelum sempat unggah.
    unggahan = st.file_uploader("📸 Foto papan + bayangan (file ASLI kamera, "
                                "bukan hasil kompresi WhatsApp)",
                                type=["jpg", "jpeg", "png"])
    if unggahan is None:
        st.info("Unggah foto untuk melanjutkan — kalau file asli kamera (bukan "
                 "hasil forward WhatsApp), lintang/bujur/tanggal/jam di bawah "
                 "akan dicoba diisi otomatis dari metadata foto.")
        return

    if st.session_state.get("ukur_nama_file") != unggahan.name:
        st.session_state["ukur_nama_file"] = unggahan.name
        _reset_klik()
        _isi_otomatis_dari_exif(unggahan.getvalue())

    pesan_exif = st.session_state.pop("ukur_exif_pesan", None)
    if pesan_exif:
        {"success": st.success, "info": st.info}[pesan_exif[0]](pesan_exif[1])

    arr = np.frombuffer(unggahan.getvalue(), np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        st.error("Berkas tidak terbaca sebagai gambar.")
        return

    # --- 2. Gerbang mutu otomatis (cek foto dulu sebelum minta apa-apa lagi) ---
    hasil_mutu = pc.cek_foto_ukur(img_bgr, verbose=False)
    vonis = hasil_mutu.get("vonis", "ULANGI")
    warna = {"AMAN": st.success, "SEDANG": st.warning, "ULANGI": st.error}[vonis]
    warna(f"**{vonis}** — {hasil_mutu.get('alasan', '')}")
    if "bias_deg" in hasil_mutu:
        st.caption(f"Bias sudut perkiraan: {hasil_mutu['bias_deg']:.3f}° · "
                   f"papan mengisi {hasil_mutu['isi_frame']*100:.0f}% lebar frame · "
                   f"{hasil_mutu.get('sumber','')}")
    if vonis == "ULANGI":
        st.stop()

    # --- 3. Waktu potret (kalau ada di EXIF, sudah otomatis dari langkah 1;
    # kalau tidak ada, isi manual di sini) ---
    st.divider()
    _terapkan_exif_pending_jam()  # sebelum widget ukur_hm/ukur_sec dibuat
    tz = dt.timezone(dt.timedelta(hours=inp["tz_offset"]))
    is_today = inp["the_date"] == user_today(inp["tz_offset"])
    if "ukur_hm" not in st.session_state:
        now_t = dt.datetime.now(tz).time() if is_today else dt.time(12, 0, 0)
        st.session_state["ukur_hm"] = now_t.replace(second=0, microsecond=0)
        st.session_state["ukur_sec"] = now_t.second if is_today else 0

    def _set_now_ukur():
        now_t2 = dt.datetime.now(tz).time()
        st.session_state["ukur_hm"] = now_t2.replace(second=0, microsecond=0)
        st.session_state["ukur_sec"] = now_t2.second

    c1, c2, c3 = st.columns([2, 1, 1])
    c1.time_input("⏱️ Waktu potret", key="ukur_hm", step=60)
    c2.number_input("Detik", 0, 59, key="ukur_sec")
    c3.button("🔄 Sekarang", width="stretch", help="Isi waktu saat ini",
             on_click=_set_now_ukur, key="btn_now_ukur")

    t = st.session_state["ukur_hm"].replace(second=int(st.session_state["ukur_sec"]))
    dt_local = dt.datetime.combine(inp["the_date"], t, tzinfo=tz)

    ensure_ephemeris()
    try:
        out = qc.solve_instant(inp["lat"], inp["lon"], inp["elev"], dt_local, inp["height"])
    except Exception as e:  # noqa: BLE001
        st.error(f"Kesalahan perhitungan: {e}")
        return
    a_k, sol, arah = out["a_k"], out["solution"], out["arah"]

    if sol.sun_altitude <= 0:
        st.warning("☾ Matahari di bawah ufuk pada waktu potret ini — tidak ada "
                   "bayangan. Perbaiki jam potret di atas.")
        return

    d1, d2 = st.columns(2)
    d1.metric("ΔA target (dari perhitungan)", f"{sol.delta_a:.3f}° ke {arah.upper()}")
    d2.metric("Altitude Matahari saat itu", qc.decimal_to_dms(sol.sun_altitude, "alt"))

    # Peringatan kualitas altitude — informasional di alur foto (fotonya sudah
    # diambil), tapi tetap berguna sebagai sinyal seberapa jauh dipercaya hasil
    # §7.3 di bawah, dan pengingat untuk sesi pemotretan berikutnya.
    if sol.sun_altitude < 10:
        st.warning("Altitude rendah: bayangan panjang & ujung kabur (penumbra "
                   "melebar). Pertimbangkan waktu lain untuk sesi berikutnya.")
    elif sol.sun_altitude > 65:
        st.warning(
            f"Altitude tinggi ({sol.sun_altitude:.0f}°): bayangan akan pendek — "
            "juga memperbesar dampak kemiringan gnomon (galat ≈ kemiringan × "
            "tan(altitude)). Kalau bisa, pilih jam dengan altitude 15°-60° "
            "untuk sesi berikutnya."
        )
    else:
        st.success("Altitude cukup ideal untuk pengukuran bayangan (±15°–60°).")

    # --- 4. Data alat ---
    st.divider()
    e1, e2 = st.columns(2)
    tinggi_gnomon = e1.number_input("Tinggi gnomon di papan (meter)", min_value=0.01,
                                    value=0.10, step=0.01, format="%.3f")
    kotak_mm = e2.number_input("Sisi kotak papan — HASIL UKUR nyata (mm)",
                               min_value=1.0, value=float(pc.SQUARE_MM), step=0.1,
                               help="Ukur dengan penggaris/jangka sorong setelah "
                                    "dicetak. Jangan biarkan nilai bawaan kalau "
                                    "cetakan tidak 100% actual size.")

    # --- 5. Rektifikasi ortho + penandaan titik ---
    st.divider()
    _, board, _, det = pc.build_detector()
    cc, ci = hasil_mutu["_cc"], hasil_mutu["_ci"]
    try:
        ortho_bgr, meta = pc.rectify_ortho(img_bgr, cc, ci, board)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    px_per_mm = meta["px_per_mm"]
    r_marker = max(int(px_per_mm * 2), 4)
    O_px = pc.mm_ke_px_ortho(np.array(pc.GNOMON_TETAP_MM, dtype=float), meta)

    def _tampilan(dgn_B=None):
        vis = ortho_bgr.copy()
        cv2.circle(vis, tuple(O_px.astype(int)), r_marker, (0, 0, 220), 2)
        cv2.drawMarker(vis, tuple(O_px.astype(int)), (0, 0, 220),
                       cv2.MARKER_CROSS, r_marker * 2, 2)
        if dgn_B is not None:
            cv2.circle(vis, tuple(np.array(dgn_B).astype(int)), r_marker, (0, 140, 0), -1)
        return Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))

    if "ukur_titik_B" not in st.session_state:
        st.info("🖱️ Titik **O** (tanda silang biru) sudah otomatis di posisi gnomon "
                "tetap. Klik **ujung bayangan** (titik terjauh bayangan gnomon) "
                "pada gambar di bawah — atau coba tombol deteksi otomatis dahulu.")

        # --- Tahap 1 (eksperimental): deteksi otomatis, klik tetap jadi koreksi ---
        colb1, colb2 = st.columns([1, 2])
        if colb1.button("🤖 Coba deteksi otomatis", key="btn_auto_B"):
            auto = pc.deteksi_otomatis_bayangan(ortho_bgr, meta, np.array(pc.GNOMON_TETAP_MM, dtype=float))
            if not auto["yakin"]:
                colb2.error("Tidak ketemu objek gelap yang menempel di titik O — "
                            "klik manual di bawah.")
            else:
                st.session_state["ukur_titik_B"] = auto["tip_px"]
                if auto["pakai_warna"]:
                    st.session_state["ukur_auto_pesan"] = (
                        "success",
                        f"Gnomon berwarna terdeteksi ({auto['n_piksel_gnomon']} px) dan "
                        f"disingkirkan — ujung bayangan diambil dari sisa piksel netral "
                        f"({auto['n_piksel_bayangan']} px)."
                    )
                else:
                    st.session_state["ukur_auto_pesan"] = (
                        "warning",
                        "Gnomon TIDAK terdeteksi berwarna (foto lama / gnomon polos) — "
                        "titik diambil dari ujung terjauh objek gelap apa adanya "
                        "(gnomon+bayangan tercampur). Periksa & koreksi manual kalau "
                        "meleset."
                    )
                st.rerun()
        colb2.caption("Eksperimental — masih perlu diperiksa manual. Kerja lebih baik "
                      "kalau gnomon dicat warna marun/magenta TUA (lihat "
                      "`papan_charuco.GNOMON_HSV_LO/HI`).")

        koor = streamlit_image_coordinates(_tampilan(), key="klik_B")
        if koor is not None:
            st.session_state["ukur_titik_B"] = (koor["x"], koor["y"])
            st.session_state.pop("ukur_auto_pesan", None)
            st.rerun()
        return

    pesan_auto = st.session_state.pop("ukur_auto_pesan", None)
    if pesan_auto:
        {"success": st.success, "warning": st.warning}[pesan_auto[0]](pesan_auto[1])

    # --- 6. Hasil ---
    B_px = np.array(st.session_state["ukur_titik_B"], dtype=float)
    # Skala ortho dihitung dari SQUARE_MM bawaan; kalau ukuran kotak hasil ukur
    # nyata beda, seluruh jarak metrik (mm) di ortho ikut dikoreksi rasio ini.
    rasio_skala = kotak_mm / pc.SQUARE_MM
    O_mm = np.array(pc.GNOMON_TETAP_MM, dtype=float) * rasio_skala
    B_mm = pc.px_ortho_ke_mm(B_px, meta) * rasio_skala

    hasil = pc.hitung_pengukuran(O_mm, B_mm, sol.delta_a, arah,
                                 tinggi_gnomon, sol.sun_altitude)

    # gambar overlay memakai koordinat ortho ASLI (bukan yang sudah dikoreksi
    # rasio), supaya posisi piksel di layar tetap tepat; skala rasio hanya
    # memengaruhi ANGKA (mm, altitude), bukan gambar.
    O_mm_disp = np.array(pc.GNOMON_TETAP_MM, dtype=float)
    B_mm_disp = pc.px_ortho_ke_mm(B_px, meta)
    vis = pc.gambar_overlay(ortho_bgr, meta, O_mm_disp, B_mm_disp,
                            hasil["v_kiblat_mm"], sol.delta_a, arah)
    st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB),
            caption="Garis merah = bayangan terukur · garis hijau = arah Kiblat",
            width="stretch")

    f1, f2, f3 = st.columns(3)
    f1.metric("Panjang bayangan", f"{hasil['panjang_bayangan_mm']:.1f} mm")
    f2.metric("Altitude terukur (§7.3)", f"{hasil['altitude_terukur_deg']:.2f}°",
             delta=f"{hasil['selisih_altitude_deg']:+.2f}° vs efemeris")
    f3.metric("Validasi altitude", "✅ OK" if hasil["validasi_altitude_ok"] else "⚠️ Periksa")

    # --- 6a. Petunjuk Pemasangan Fisik: baca satu angka di tepi papan ---
    st.divider()
    st.subheader("📐 Petunjuk Pemasangan Fisik")
    potong = pc.titik_potong_tepi(O_mm_disp, hasil["v_kiblat_mm"])
    if potong["sisi"] is None:
        st.warning("Garis Kiblat tidak memotong tepi papan (arahnya nyaris "
                   "sejajar tepi) — perbesar papan atau geser titik pengamatan.")
    else:
        st.success(
            f"Baca di tepi **{potong['sisi'].upper()}** papan, pada angka "
            f"**{potong['posisi_cm_str']} cm** dari pojok kiri-atas pola. "
            "Tandai titik itu, lalu rentangkan tali kencang atau laser dari "
            "situ melalui titik O — itulah garis Kiblat fisik, siap "
            "diperpanjang sejauh yang dibutuhkan."
        )
        # crop-zoom di sekitar titik potong (dari citra ortho, bukan sekadar
        # skema) supaya pengguna tinggal mencocokkan visual, bukan menghitung
        # interpolasi sendiri antar goresan mm.
        titik_potong_px = pc.mm_ke_px_ortho(potong["titik_mm"], meta)
        zoom = pc.crop_zoom_titik(vis, meta, titik_potong_px, radius_px=70, skala_output=5)
        st.image(cv2.cvtColor(zoom, cv2.COLOR_BGR2RGB),
                caption=f"Perbesaran di sekitar titik potong ({potong['posisi_cm_str']} cm) "
                        "— cocokkan tanda silang dengan goresan di papan fisik.",
                width=380)

    if not hasil["validasi_altitude_ok"]:
        st.warning(
            "Selisih altitude terukur vs efemeris melebihi ambang 0,5° — "
            "kemungkinan papan tidak rata, gnomon tidak tegak, tinggi gnomon "
            "salah diinput, atau kedua titik salah tandai. Periksa sebelum "
            "memakai hasil ini."
        )

    # --- 6b. Opsional: bandingkan ke garis yang sudah terpasang ---
    with st.expander("↔️ Bandingkan ke garis yang SUDAH ADA di lokasi (opsional)"):
        st.caption(
            "Kalau di lokasi ini sudah ada tanda arah (garis shaf, tanda kiblat "
            "lama, dsb.) yang ikut terlihat di foto, klik satu titik di "
            "sepanjang tanda itu (titik kedua dianggap sama dengan pangkal "
            "gnomon O) untuk mengetahui berapa derajat koreksinya."
        )
        koor_ref = streamlit_image_coordinates(
            _tampilan(dgn_B=B_px), key="klik_ref")
        if koor_ref is not None:
            st.session_state["ukur_titik_ref"] = (koor_ref["x"], koor_ref["y"])
        ref_px = st.session_state.get("ukur_titik_ref")
        selisih_ref = None
        if ref_px is not None:
            R_mm = pc.px_ortho_ke_mm(np.array(ref_px, dtype=float), meta)
            v_ref = R_mm - O_mm_disp
            v_kib = hasil["v_kiblat_mm"]
            ang_ref = math.degrees(math.atan2(v_ref[1], v_ref[0]))
            ang_kib = math.degrees(math.atan2(v_kib[1], v_kib[0]))
            selisih_ref = ((ang_ref - ang_kib + 180) % 360) - 180
            arah_koreksi = "KIRI" if selisih_ref > 0 else "KANAN"
            st.metric("Garis terpasang meleset dari Kiblat",
                     f"{abs(selisih_ref):.2f}° — putar ke {arah_koreksi}")

    # --- 7. Simpan & log ---
    st.divider()
    catatan = st.text_input("Catatan (opsional)", "")
    if st.button("💾 Simpan pengukuran ini ke log", type="primary"):
        _catat_log_ukur({
            "waktu_dicatat": dt.datetime.now(tz).isoformat(timespec="seconds"),
            "nama_file": unggahan.name,
            "lat": f"{inp['lat']:.6f}", "lon": f"{inp['lon']:.6f}",
            "waktu_potret": dt_local.isoformat(timespec="seconds"),
            "tinggi_gnomon_m": f"{tinggi_gnomon:.3f}",
            "kotak_mm": f"{kotak_mm:.2f}",
            "panjang_bayangan_mm": f"{hasil['panjang_bayangan_mm']:.1f}",
            "delta_a_target_deg": f"{sol.delta_a:.3f}",
            "arah": arah,
            "altitude_efemeris_deg": f"{sol.sun_altitude:.3f}",
            "altitude_terukur_deg": f"{hasil['altitude_terukur_deg']:.3f}",
            "selisih_altitude_deg": f"{hasil['selisih_altitude_deg']:.3f}",
            "validasi_altitude_ok": hasil["validasi_altitude_ok"],
            "vonis_mutu_foto": vonis,
            "bias_mutu_deg": f"{hasil_mutu.get('bias_deg', float('nan')):.3f}"
                            if "bias_deg" in hasil_mutu else "",
            "sisi_potong_tepi": potong.get("sisi") or "",
            "posisi_potong_cm": potong.get("posisi_cm_str", ""),
            "selisih_referensi_deg": (f"{selisih_ref:.3f}"
                                      if selisih_ref is not None else ""),
            "catatan": catatan,
        })
        st.success(f"Tersimpan ke {LOG_UKUR_CV}")

    if st.button("🔁 Ukur foto lain / ulangi penandaan"):
        _reset_klik()
        st.rerun()


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

    # Mode dipilih DULU, sebelum sidebar dibangun — supaya sidebar tahu field
    # mana yang relevan ditampilkan untuk mode ini (lihat sidebar_inputs()).
    mode = st.radio(
        "Mode pengukuran",
        ["🗓️ Terjadwal (ΔA bulat — busur manual)",
         "🎯 Bebas waktu (ΔA presisi — kamera/CV)",
         "📷 Ukur dari Foto (ChArUco)"],
        horizontal=True,
    )

    inp = sidebar_inputs(mode)

    # Penjaga akurasi: kecocokan zona waktu dengan bujur
    expected = inp["lon"] / 15.0
    if abs(inp["tz_offset"] - expected) > 1.5:
        st.warning(f"⚠️ Zona waktu (UTC{inp['tz_offset']:+g}) tampak tak cocok dengan bujur "
                   f"{inp['lon']:.2f}° (perkiraan UTC{expected:+.1f}). Pastikan zona waktu benar — "
                   "salah zona menggeser jam eksekusi.")

    st.caption("🕐 Waktu eksekusi mengacu ke jam resmi **BMKG** (diselaraskan otomatis saat "
               "online). Jika offline, gunakan jam perangkat mode otomatis.")

    if mode.startswith("🎯"):
        st.metric("Azimuth Kiblat lokasi ini (A_k)",
                  qc.decimal_to_dms(qc.azimuth_kiblat(inp["lat"], inp["lon"]), "az"))
        st.divider()
        render_instant_mode(inp, get_time_reference())
        return

    if mode.startswith("📷"):
        st.metric("Azimuth Kiblat lokasi ini (A_k)",
                  qc.decimal_to_dms(qc.azimuth_kiblat(inp["lat"], inp["lon"]), "az"))
        st.divider()
        render_ukur_foto(inp, get_time_reference())
        return

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
