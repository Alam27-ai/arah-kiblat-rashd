# Rencana Pengembangan: Penentuan Arah Kiblat Otomatis Berbasis Computer Vision

**Dokumen kerja — BayangKiblat**
Disusun: 2 Agustus 2026

Dokumen ini menyusun rencana penambahan modul *computer vision* (CV) pada
BayangKiblat: mengukur arah bayangan gnomon dari foto secara otomatis, sehingga
pembacaan busur derajat oleh mata manusia — yang selama ini menjadi mata rantai
paling lemah — dapat digantikan pengukuran citra ber-presisi sub-piksel.

---

## 1. Posisi terhadap literatur (jujur, hasil penelusuran)

Sebelum menyusun rencana, dilakukan penelusuran untuk memastikan apa yang sudah
ada dan apa yang belum. Ringkasannya:

**Yang SUDAH ada (jangan diklaim baru):**

| Sudah ada | Sumber |
|---|---|
| Rashdul Qiblat lokal (harian) & global | Slamet Hambali; Lestari & Ramadhan (2024) |
| Metode bayangan pada sudut "mudah diukur" (0°, 90°, 180°, 270°) | Abdali (1997), §5.3 *Shadow Method* |
| Software penghitung waktu-waktu tersebut | Abdali, program *Minaret* (1990-an); QiblaTime; Program Hisab Astronomis (PERSIS) |
| Toleransi ±5 menit / ±2 hari untuk Rashdul | Raharto & Surya (2011); Hadi Bashori (2015) |
| Estimasi posisi Matahari **dari** bayangan via CV | Junejo & Foroosh, *GPS coordinates estimation and camera calibration from solar shadows* (CVIU 2010); Sun-azimuth forensics (IEEE TIFS 2012) |
| Deteksi tepi sub-piksel | Devernay; Grompone von Gioi & Randall (IPOL 2017) |

**Yang belum ditemukan padanannya (kandidat kontribusi):**

Literatur CV yang ada memecahkan **masalah kebalikan**: dari bayangan →
menduga posisi Matahari/lokasi/waktu (untuk forensik & geolokasi). Kasus kita
adalah **masalah maju** dan jauh lebih menguntungkan: lokasi dan waktu sudah
diketahui presisi (GPS + efemeris DE440s + jam BMKG), sehingga azimuth Matahari
**sudah diketahui sebagai kebenaran acuan**. Yang perlu diukur dari citra
hanyalah *sudut relatif* antara garis bayangan dan garis Kiblat pada satu bidang
— tanpa perlu kompas, tanpa perlu Utara sejati.

> **Catatan kejujuran:** penelusuran ini belum sistematis (belum menyisir basis
> data Scopus/WoS penuh). Sebelum diklaim orisinal dalam tulisan ilmiah, wajib
> dilakukan tinjauan pustaka formal.

---

## 2. Gagasan inti

Gnomon tegak, permukaan datar, papan kalibrasi ChArUco diletakkan di bidang ukur.
Pengguna memotret. Aplikasi sudah tahu azimuth Matahari `A_s` pada detik itu,
maka azimuth bayangan `= A_s + 180°`. Jadi **garis bayangan pada citra adalah
acuan sudut yang azimuth sejatinya sudah diketahui**. Aplikasi tinggal:

1. merektifikasi citra ke tampak-atas (homografi dari papan ChArUco);
2. mengukur garis bayangan dengan presisi sub-piksel;
3. memutar `ΔA` dari garis itu → menggambar garis Kiblat di atas foto.

Utara sejati tidak pernah dibutuhkan. Busur derajat fisik tidak dibutuhkan.

---

## 3. Metode CV yang dipilih (dan alasannya)

Untuk tugas **metrologi** (mengukur, bukan mengenali), metode klasik berbasis
model lebih akurat dan lebih dapat dipertanggungjawabkan daripada *deep
learning*. Rancangan yang disarankan bersifat hibrida:

| Tahap | Metode | Alasan |
|---|---|---|
| Deteksi kasar bayangan | Segmentasi (klasik/DL ringan) | Hanya untuk menentukan ROI, tidak menentukan akurasi |
| Kalibrasi kamera | Model Brown–Conrady (koef. distorsi radial+tangensial) | Wajib: distorsi lensa HP membengkokkan garis lurus |
| Rektifikasi bidang | Homografi dari papan **ChArUco** | Sudut hanya sahih diukur pada bidang yang sudah direktifikasi |
| **Lokalisasi tepi** | **Fitting profil tepi (ESF/erf) sub-piksel**, bukan Canny biner | Inti akurasi — lihat §4 |
| Estimasi garis | *Total least squares* pada ratusan potongan melintang | Rata-rata banyak titik menekan derau |
| Verifikasi ketegakan | Fotogrametri *vanishing point* | Memverifikasi asumsi, tidak sekadar mempercayainya |

**Mengapa bukan Canny/Hough biasa:** penumbra membuat tepi bayangan *melebar
secara fisis* (bukan cacat citra). Detektor tepi biner membuang justru informasi
yang paling berharga. Devernay mencapai ~0,05 px pada citra ber-SNR tinggi;
namun untuk tepi yang sengaja kabur, pendekatan yang tepat adalah **memodelkan
seluruh profil kecerahan** dan mencari titik-tengahnya.

---

## 4. Argumen ilmiah utama: penumbra adalah batas *presisi*, bukan batas *akurasi*

Ini landasan yang membuat gagasan ini layak diteliti.

Piringan Matahari berdiameter sudut ~0,53°, sehingga tepi bayangan pada jarak
`d` dari gnomon melebar selebar `d·tan(0,53°)`. Untuk bayangan 1,2 m, lebarnya
**~1,1 cm** — inilah alasan mata manusia sulit menandai "ujung bayangan", dan
alasan literatur menyebut penumbra sebagai batas fisik metode bayangan.

Tetapi: profil kecerahan penumbra itu **simetris** (piringan Matahari simetris
radial; *limb darkening* pun radial simetris). Titik-tengah profil karena itu
merupakan penduga **tak-bias** terhadap tepi bayangan geometris. Artinya
penumbra membatasi *presisi pembacaan mata*, bukan *akurasi metode*. CV dapat
memanfaatkan seluruh gradien (puluhan piksel informasi per potongan, ratusan
potongan sepanjang bayangan) untuk menemukan titik-tengah itu jauh lebih presisi
daripada mata.

> **Hipotesis yang bisa diuji:** apakah titik yang biasa ditandai praktikan
> (tepi yang "tampak") mengandung **bias sistematis** terhadap tepi geometris?
> Bila ya, besarnya berapa dan dapatkah dikoreksi? Ini pertanyaan riset yang
> belum terjawab di literatur Falak.

**Simulasi Monte Carlo** (gnomon 1 m, altitude 40°, citra 4000 px, papan 1,5 m):

| SNR citra | Presisi 1 tepi | Presisi sudut (200 potongan) |
|---|---|---|
| 50 | 3,3 px (1,23 mm) | 0,0145° |
| 100 | 2,6 px (0,97 mm) | 0,0114° |
| 200 | 1,8 px (0,68 mm) | 0,0080° |

Bandingkan pembacaan busur derajat manual: ±0,5°–1°. Selisihnya **satu-dua orde
besaran**.

---

## 5. Temuan penting: kemiringan gnomon diperkuat oleh tan(altitude)

Diturunkan dan diverifikasi numerik dalam pengerjaan dokumen ini:

```
galat_azimuth  ≈  kemiringan_gnomon × tan(altitude Matahari)
```

| Altitude | tilt 0,5° | tilt 1° | tilt 2° |
|---|---|---|---|
| 15° | 0,134° | 0,268° | 0,536° |
| 30° | 0,289° | 0,577° | 1,155° |
| 45° | 0,500° | 1,000° | 2,000° |
| 60° | 0,866° | 1,732° | 3,466° |
| 70° | 1,374° | 2,747° | 5,505° |

**Implikasi yang berlawanan dengan anjuran umum.** Panduan Falak lazim
menyarankan altitude 15°–60° tanpa turunan kuantitatif. Tabel di atas
menunjukkan Matahari **tinggi justru berbahaya**: pada 70°, kemiringan gnomon 1°
saja (mudah terjadi) sudah menghasilkan galat 2,7°.

Sebaliknya, kemiringan **bidang papan** (tidak waterpass) ternyata berdampak
jauh lebih kecil pada pengukuran sudut *di dalam bidang* (~0,002° per 1°
kemiringan) — temuan yang perlu diverifikasi lebih lanjut, tetapi bila benar,
berarti prioritas di lapangan seharusnya **ketegakan gnomon**, bukan kerataan
papan seperti yang biasa ditekankan.

**Sensitivitas waktu** (Pekalongan, 30 Juli 2026):

| Jam | Altitude | Laju azimuth | Galat bila telat 5 detik |
|---|---|---|---|
| 07:00 | 14,7° | 0,054°/menit | 0,0045° |
| 09:00 | 41,5° | 0,148°/menit | 0,0123° |
| 12:00 | 64,4° | 0,545°/menit | **0,0454°** |
| 15:00 | 36,3° | 0,118°/menit | 0,0098° |
| 17:00 | 9,0° | 0,043°/menit | 0,0036° |

Dekat tengah hari, ketepatan waktu jadi 10× lebih kritis — alasan tambahan
menghindari altitude tinggi.

---

## 6. Anggaran galat gabungan: manual vs CV

Asumsi: gnomon 1 m; ketidaktegakan 1° (unting-unting manual) vs 0,2° (dengan
verifikasi CV); baca busur 0,5° vs fit garis 0,02°; penandaan waktu 10 s vs 2 s.

| Altitude | Panjang bayangan | Total MANUAL | Total CV | Perbaikan |
|---|---|---|---|---|
| 15° | 3,73 m | 0,568° | 0,057° | 9,9× |
| 20° | 2,75 m | 0,619° | 0,076° | 8,2× |
| 25° | 2,14 m | 0,684° | 0,095° | 7,2× |
| 30° | 1,73 m | 0,764° | 0,117° | 6,5× |
| 40° | 1,19 m | 0,977° | 0,169° | 5,8× |
| 45° | 1,00 m | 1,118° | 0,201° | 5,6× |
| 60° | 0,58 m | 1,803° | 0,347° | 5,2× |
| 70° | 0,36 m | 2,793° | 0,550° | 5,1× |

**Rekomendasi operasional: altitude 20°–35°** (bayangan 1,4–2,7 m) — kompromi
antara penguatan `tan(altitude)` dan kebutuhan bidang datar yang luas.

Perhatikan: setelah CV dipakai, **penyempit utama bukan lagi pembacaan sudut,
melainkan ketegakan gnomon**. Karena itu verifikasi ketegakan berbasis citra
menjadi komponen wajib, bukan pelengkap.

---

## 7. Rencana bertahap

### Tahap 0 — Persiapan (langsung bisa dikerjakan)
- Tambah `st.camera_input()` untuk dokumentasi foto pada laporan PDF.
- Tambah label tingkat keandalan sudut pada tabel peluang ΔA
  (0°/180° tanpa alat · 90° siku · 45° lipatan kertas · lainnya butuh busur).
- Nilai: langsung berguna, nol risiko, tanpa dependensi baru.

### Tahap 1 — Prototipe pengukuran (inti teknis)
- Cetak papan ChArUco; skrip kalibrasi kamera (OpenCV).
- Pipeline: undistort → deteksi ChArUco → homografi → ROI bayangan →
  fitting profil tepi sub-piksel → TLS garis → sudut.
- Uji pada foto sintetis (kebenaran acuan diketahui) sebelum foto nyata.

### Tahap 2 — Validasi lapangan
- Bandingkan CV vs busur manual vs **theodolit** (acuan) pada ≥30 pengukuran,
  beragam altitude (15°–60°), beragam kamera HP dan kondisi cahaya.
- Analisis: bias, simpangan baku, dekomposisi sumber galat, Bland–Altman.
- Ini bagian yang mengubah proyek rekayasa menjadi **penelitian**.

### Tahap 3 — Integrasi aplikasi
- Alur: pilih waktu → hitung mundur → jepret → hasil sudut + garis Kiblat
  ditumpangkan pada foto → masuk laporan PDF.
- Peringatan otomatis bila ketegakan gnomon gagal diverifikasi.

### Tahap 4 — Publikasi (opsional)
Kandidat kontribusi, diurutkan dari yang paling kuat:
1. Model bias penumbra + koreksinya (optika + CV).
2. Anggaran galat formal metode bayangan Kiblat mengikuti kerangka **GUM**
   (ISO/BIPM) — belum pernah diterapkan pada ranah ini sejauh penelusuran.
3. Turunan `τ·tan(a)` dan implikasinya terhadap pemilihan altitude optimal —
   mengoreksi anjuran kualitatif yang lazim di literatur Falak.

---

## 8. Risiko dan batasan (jujur)

- **Ini proyek nyata, bukan tempelan fitur.** Tahap 1–2 realistis memakan
  waktu berbulan-bulan, bukan sore-ini-jadi-besok.
- Semua angka di dokumen ini berasal dari **simulasi**, bukan foto asli.
  Kamera HP nyata membawa masalah tambahan: kompresi JPEG, HDR agresif,
  *rolling shutter*, penajaman otomatis yang merusak profil tepi. **Foto RAW
  sangat disarankan** untuk tahap pengukuran.
- Klaim "belum ada yang melakukan" masih perlu tinjauan pustaka formal.
- Nilai praktis di lapangan tetap dibatasi kondisi nyata: langit cerah,
  permukaan datar, dan kesabaran praktikan.

---

## Pustaka

- Abdali, S. K. (1997). *The Correct Qibla*. — §5.3 Shadow Method.
- Grompone von Gioi, R. & Randall, G. (2017). *A Sub-Pixel Edge Detector:
  an Implementation of the Canny/Devernay Algorithm*. IPOL.
- Junejo, I. N. & Foroosh, H. (2010). *GPS coordinates estimation and camera
  calibration from solar shadows*. Computer Vision and Image Understanding.
- Lestari, T. & Ramadhan, R. (2024). *Peran Penting Posisi Matahari dalam
  Penentuan Rashdul Qiblat Lokal dan Global*. ELFALAKY, 8(1).
- Raharto, M. & Surya, D. J. A. (2011). *Telaah Penentuan Arah Kiblat dengan
  Perhitungan Trigonometri Bola dan Bayang-Bayang Gnomon oleh Matahari*.
  Jurnal Fisika HFI, 11(1), 23–29.
- Hadi Bashori, M. (2015). *Pengantar Ilmu Falak*. Pustaka Al Kautsar.
- BIPM/ISO. *Guide to the Expression of Uncertainty in Measurement* (GUM).
