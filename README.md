# BayangKiblat — Penentu Arah Kiblat berbasis Matahari

Aplikasi web interaktif (Streamlit) untuk menentukan **arah Kiblat** di lokasi
mana pun di Indonesia menggunakan **bayangan Matahari**, berbasis efemeris
presisi tinggi NASA/JPL. Ditujukan sebagai media praktikum mahasiswa Ilmu Falak
sekaligus alat bantu lapangan yang akurat.

Aplikasi menyajikan dua pendekatan yang saling melengkapi:

1. **Rashdul Qiblah harian** (ΔA = 0) — momen ketika Matahari tepat *searah* atau
   *berlawanan* arah Kiblat, sehingga bayangan tongkat **langsung** menjadi garis
   Kiblat tanpa alat bantu busur.
2. **Metode Selisih Azimuth Matahari Harian** (ΔA ≠ 0) — untuk hampir setiap saat
   di siang hari: aplikasi menghitung sudut busur (ΔA) yang harus diputar dari
   garis bayangan untuk memperoleh arah Kiblat, lengkap dengan waktu eksekusi
   presisi detik.

---

## Fitur

- Masukan lokasi cepat: **daftar kota Indonesia**, **tempel koordinat Google Maps**,
  GPS perangkat (opsional), atau **derajat-menit-detik** manual.
- Perhitungan **semua peluang** pada satu hari (pagi & sore), terurut waktu,
  dapat dipilih per waktu.
- **Rashdul Qiblah harian** (searah & sebaliknya 180°).
- **Simulasi** tampak samping (tongkat–bayangan) dan tampak atas (kompas + busur → Kiblat).
- **Hitung mundur beraba-aba suara** menuju detik eksekusi, **diselaraskan ke jam
  resmi BMKG** (NTP sisi server) dengan koreksi jam perangkat; fallback HTTP & jam lokal.
- Keluaran azimuth dalam **derajat-menit-detik**; **laporan ringkas** siap cetak.
- **Penjaga akurasi**: peringatan zona waktu vs bujur, dan selisih jam perangkat vs BMKG.

---

## Landasan Ilmiah dan Metode

### 1. Azimuth Kiblat (Aturan Trigonometri Bola / Great Circle)

Azimuth Kiblat sejati `A_k` (diukur dari Utara sejati, searah jarum jam) dihitung
dengan rumus *initial bearing* lingkaran besar dari pengamat (φ, λ) menuju Kakbah
(φ_K, λ_K):

```
Δλ  = λ_K − λ
A_k = atan2( sin Δλ ,  cos φ · tan φ_K − sin φ · cos Δλ )   (mod 360°)
```

### 2. Target Azimuth Matahari (metode selisih azimuth)

Pada garis bayangan tongkat tegak, azimuth bayangan = azimuth Matahari + 180°.
Untuk memperoleh Kiblat dengan memutar busur sebesar ΔA dari garis bayangan:

```
Sesi Pagi : A_s = (A_k − 180° − ΔA) mod 360°   → putar ΔA ke KANAN (searah jarum jam)
Sesi Sore : A_s = (A_k + ΔA − 180°) mod 360°   → putar ΔA ke KIRI  (berlawanan jarum jam)
```

di mana `A_s` adalah azimuth Matahari yang dituju. Aplikasi mencari **waktu** saat
azimuth Matahari sama dengan `A_s`.

### 3. Rashdul Qiblah harian (kasus ΔA = 0)

```
Searah      : azimuth Matahari = A_k          → bayangan berlawanan Kiblat
Sebaliknya  : azimuth Matahari = A_k + 180°   → bayangan tepat menunjuk Kiblat
```

### 4. Panjang bayangan

```
L = tinggi_tongkat / tan(altitude Matahari)
```

### 5. Pencari waktu (solver)

Karena azimuth Matahari **tidak monoton** di daerah tropis (khususnya mendekati
tanggal kulminasi zenith), solver memakai **pemindaian kasar + bisection** dengan
*pembungkusan* selisih sudut ke rentang (−180°, 180°], sehingga tahan terhadap
gerak semu non-monoton dan kasus dua perpotongan. Waktu diselesaikan hingga
tingkat detik.

---

## Sumber Data dan Akurasi

- **Efemeris**: NASA/JPL Development Ephemeris **DE440s** melalui pustaka
  [Skyfield](https://rhodesmill.org/skyfield/). Posisi Matahari memakai posisi
  *apparent* (koreksi light-time & aberasi).
- **Koordinat Kakbah** (acuan resmi Kemenag RI):
  21° 25′ 21,17″ LU ; 39° 49′ 34,56″ BT → 21,422547° ; 39,826267°.
- **Refraksi atmosfer** diabaikan untuk *azimuth* (pengaruhnya tak berarti pada
  azimuth); prediksi *panjang bayangan* bersifat geometris sehingga saat Matahari
  sangat rendah nilainya sedikit lebih panjang daripada kenyataan.
- **Akurasi lapangan**: presisi hitungan jauh melampaui kebutuhan; galat akhir di
  lapangan didominasi oleh ketegakan tongkat, kerataan papan (waterpass),
  ketajaman ujung bayangan, dan pembacaan busur — realistis beberapa menit busur.
  Karena itu ΔA dibulatkan ke derajat bulat (menjaga akurasi pembacaan busur),
  sementara ketepatan waktu ditangani oleh hitung mundur.

---

## Instalasi dan Menjalankan

Prasyarat: Python 3.10+.

```bash
pip install -r requirements.txt
streamlit run app.py
```

Efemeris **`de440s.bsp` (±32 MB)** disertakan di repositori (di-commit via Git
Bash/CLI) sehingga aplikasi berjalan tanpa perlu mengunduh. Catatan: berkas ini
tidak bisa diunggah lewat web GitHub (batas 25 MB/berkas) — gunakan Git Bash
(lihat `DEPLOY.md`). Bila berkas tak ada, aplikasi mengunduhnya otomatis sekali
saat ada internet. Versi penuh `de440.bsp` (±114 MB) tidak diperlukan — hasilnya
identik dengan `de440s.bsp`.

GPS perangkat bersifat opsional: `pip install streamlit-geolocation`.

---

## Struktur Proyek

```
qibla_core.py         Mesin perhitungan (tanpa Streamlit) — dapat diuji terpisah
app.py                Antarmuka Streamlit
test_qibla_core.py    Uji otomatis engine
requirements.txt      Dependensi
de440s.bsp            Efemeris JPL DE440s (~32 MB) — di-commit via Git Bash
.gitignore            mengecualikan de440s.bsp/de440.bsp (lihat catatan di dalamnya)
LICENSE               Lisensi MIT
DEPLOY.md             Panduan upload GitHub & deploy
```

## Pengujian

```bash
python test_qibla_core.py
```

Mencakup: azimuth Kiblat, konsistensi rumus ΔA, solver, kasus tanpa solusi,
`compute_all` (urut waktu, ΔA bulat, geometri busur→Kiblat), Rashdul Qiblah, dan
konversi DMS.

---

## Menerbitkan Online (Deploy)

Lihat panduan lengkap di `DEPLOY.md`. Ringkas: unggah repo ke GitHub (sertakan
`de440s.bsp` agar tak perlu unduh), lalu deploy gratis via **Streamlit Community
Cloud** (share.streamlit.io) atau **Hugging Face Spaces** (SDK Streamlit). Catatan:
banyak hosting cloud memblokir NTP (UDP) sehingga acuan waktu jatuh ke cadangan
HTTP/jam perangkat — sudah ditangani otomatis.

---

## Keterbatasan

- Metode mensyaratkan langit cerah dan permukaan benar-benar datar serta gnomon
  benar-benar tegak — aplikasi menginstruksikannya tetapi tak dapat memverifikasi.
- Prediksi panjang bayangan belum mengoreksi refraksi (bersifat ilustratif).
- Acuan waktu BMKG membutuhkan internet saat menghitung.

---

## Sitasi

Bila aplikasi ini dipakai dalam karya ilmiah, mohon sebutkan sumber data:

- Park, R. S., et al. *The JPL Planetary and Lunar Ephemerides DE440 and DE441*,
  The Astronomical Journal, 2021.
- Rhodes, B. *Skyfield: High precision research-grade positions for planets and
  Earth satellites*.
- Koordinat Kakbah: Kementerian Agama Republik Indonesia.

---

## Lisensi

MIT License — lihat berkas `LICENSE`.
