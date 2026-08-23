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

## 7. Ide pengembangan lanjutan (dikerjakan SETELAH Tahap 0–2 di §8 selesai)

> **Urutan kerja — jangan dibalik.** Sub-bagian di bawah ini (§7.1–§7.3) adalah
> **penyempurnaan presisi/keandalan**, bukan prasyarat. Baru dikerjakan setelah
> pipeline satu-frame dasar berjalan DAN sudah divalidasi ke theodolit
> (Tahap 1–2, §8) — kecuali §7.3 yang justru berguna DIKERJAKAN BERSAMAAN
> dengan Tahap 2 sebagai kontrol mutu data validasi itu sendiri. Mengerjakan
> §7.1/§7.2 lebih dulu berisiko menghabiskan waktu untuk penyempurnaan sebelum
> fondasinya sendiri terbukti benar.

### 7.1 Normalisasi latar berpola (papan ChArUco) sebelum fitting tepi bayangan

**Gagasan** (dari diskusi 2 Agustus 2026): bayangan gnomon di lapangan jatuh
melintasi papan ChArUco yang berpola kotak hitam-putih + marker — bukan latar
polos seperti asumsi literatur fitting tepi sub-piksel (Devernay, erf/sigmoid).
Kalau tepi bayangan melintasi kotak hitam vs putih, profil kecerahannya
tercampur antara efek bayangan dan efek pola papan, berisiko bias.

**Status novelty (hasil penelusuran 2 Agustus 2026):** masalah umumnya —
memisahkan iluminasi dari reflektansi pada latar bertekstur/berpola yang
**tidak diketahui** (rumput, aspal, dst.) — adalah bidang riset aktif
(*intrinsic image decomposition*) yang secara eksplisit masih diakui sebagai
"tantangan signifikan", belum tuntas. Kasus kita lebih mudah dari itu karena
polanya **diketahui persis** (papan yang kita cetak sendiri, dan homografinya
sudah wajib dihitung untuk rektifikasi geometri) — ini analog *flat-field
correction* yang mapan di astronomi/pencitraan industri. Kombinasi spesifik
"papan kalibrasi bermarker dipakai ganda: kalibrasi geometri SEKALIGUS
referensi normalisasi fotometri untuk fitting tepi bayangan sub-piksel" belum
ditemukan padanannya di penelusuran ini — kandidat kontribusi paling kuat dari
tiga ide yang dibahas 2 Agustus 2026.

**Usulan langkah teknis:**
1. Setelah homografi ChArUco dihitung (sudah wajib ada, Tahap 1), bangkitkan
   peta reflektansi yang diharapkan dari pola papan yang diketahui presisi
   (posisi & warna tiap kotak/marker dari spesifikasi cetak papan).
2. Registrasi peta itu ke citra asli lewat homografi yang sama.
3. Bagi (normalisasi) citra hasil foto dengan peta reflektansi tadi →
   hasilnya sinyal "iluminasi murni" tanpa gangguan pola papan.
4. Baru jalankan fitting erf/sigmoid sub-piksel (§3–4) di atas sinyal yang
   sudah dinormalisasi ini.

**Catatan jujur:** perlu diuji empiris — reflektansi tinta cetak & kertas
nyata tidak pernah benar-benar biner/sempurna, dan interpleksi cahaya difus
di sekitar tepi bayangan bisa menyisakan galat residual yang belum
dikuantifikasi. Ini tetap kontribusi jenis "sintesis teknik + validasi
empiris", bukan teori benar-benar baru — sama seperti §7.2 di bawah.

---

### 7.2 Burst averaging dengan constraint efemeris

**Gagasan** (dari diskusi 2 Agustus 2026): alih-alih satu foto, ambil beberapa
frame berurutan dalam jendela pendek (mis. 5 detik) dan gabungkan pembacaan
sudutnya memakai laju perubahan azimuth bayangan yang **sudah diketahui** dari
efemeris (bukan diestimasi dari citra) sebagai *constraint* saat menggabungkan
frame — mirip Kalman filter fisis, bukan rata-rata buta.

**Status novelty (jujur, hasil penelusuran 2 Agustus 2026):** prinsip umumnya
sudah mapan di tiga bidang bertetangga, jadi **tidak** bisa diklaim sebagai
teknik baru begitu saja:

| Sudah ada | Konteks |
|---|---|
| Kalman filter + efemeris untuk smoothing posisi Matahari | *Micro sun sensor* berbasis CMOS (MDPI *Sensors*, 2019) |
| Hybrid open-loop (prediksi efemeris) + closed-loop (sensor citra) | Solar tracker & heliostat — sudah puluhan tahun |
| *Shift-and-add*: geser piksel sesuai lintasan yang sudah diprediksi, lalu tumpuk | Astrometri objek Tata Surya bergerak (asteroid, TNO) |
| Rata-rata beberapa frame di sekitar titik prediksi dari model fisika | Blog pengukuran lintang via bayangan gnomon (informal, bukan jurnal) |

**Yang masih berpeluang jadi kontribusi sah:** bukan algoritmanya (itu sudah
mapan), tapi (a) penerapannya khusus untuk ekstraksi **satu sudut azimuth
statis** dari burst singkat — beda dari tracking berkelanjutan (heliostat),
deteksi objek bergerak (astrometri), atau panjang bayangan/lintang (blog di
atas) — dan (b) karakterisasi empiris seberapa besar perbaikan presisi yang
benar-benar didapat di kondisi lapangan nyata (latar bertekstur papan ChArUco,
luar ruang, bukan lab). Ini kontribusi jenis "sintesis teknik + validasi
empiris", bukan "algoritma baru" — **wajib** mengutip literatur heliostat/
sun-sensor/astrometri di atas secara eksplisit di naskah, supaya tidak
terkesan tidak tahu literatur yang relevan.

**Kaitan dengan analemma:** laju perubahan azimuth yang dipakai sebagai
constraint dihitung dari model efemeris presisi tinggi (skyfield/DE440s, sudah
dipakai `qibla_core.py`) — bukan dari rumus sundial sederhana yang butuh
koreksi *equation of time* terpisah. Karena itu efek analemma (variasi ~±16
menit antara waktu Matahari sejati dan waktu rata-rata sepanjang tahun)
**otomatis sudah benar** tanpa perlu ditangani manual — ini justru keunggulan
memakai efemeris presisi dibanding rumus sundial klasik. Analemma baru jadi
relevan secara langsung kalau teknik ini diperluas ke rata-rata **antar-hari**
(bukan cuma dalam satu burst beberapa detik) — misalnya menggabungkan data
kalibrasi dari beberapa hari berdekatan — karena laju perubahan deklinasi
Matahari berubah tidak linear di sekitar solstis. Untuk burst dalam hitungan
detik pada satu hari yang sama, efek analemma bisa dianggap konstan (sudah
"terbakukan" dalam satu angka laju azimuth saat itu), sehingga tidak perlu
koreksi tambahan di dalam jendela burst itu sendiri.

**Pustaka tambahan untuk ide ini** (belum ditinjau formal, baru hasil
pencarian web 2 Agustus 2026):
- Riset *micro sun sensor* berbasis CMOS + Kalman filter + data efemeris
  Matahari (MDPI *Sensors*, 2019 — "Accurate and Cost-Effective Micro Sun
  Sensor based on CMOS Black Sun Effect").
- Kontrol heliostat *closed-loop* berbasis citra + prediksi efemeris
  (*Solar Energy* / ScienceDirect — "Closed loop control of heliostats";
  "Novel imaging closed loop control strategy for heliostats").
- Teknik *shift-and-add* astrometri untuk objek Tata Surya bergerak dengan
  lintasan yang sudah diprediksi (arXiv, berbagai makalah survei TNO/objek
  cislunar, mis. "Optical Survey for Cislunar Moving Objects Using Image
  Stacking").

**Kapan dikerjakan:** setelah pipeline satu-frame dasar (Tahap 1–2 di bawah)
berhasil dan tervalidasi. Burst averaging adalah **penyempurnaan presisi**,
bukan prasyarat — jangan dikerjakan lebih dulu dari pipeline dasarnya.

---

### 7.3 Validasi silang altitude dari panjang bayangan (redundansi tanpa alat tambahan)

**Gagasan** (dari diskusi 2 Agustus 2026): pipeline saat ini hanya memakai
AZIMUTH bayangan (arahnya) untuk menentukan arah Kiblat. Padahal dari foto
yang sama, PANJANG bayangan bisa dipakai menghitung ALTITUDE Matahari secara
independen lewat `tan(altitude) = tinggi_gnomon / panjang_bayangan`. Altitude
ini bisa langsung dibandingkan dengan altitude yang diprediksi efemeris
(skyfield/DE440s, sudah dihitung `qibla_core.py`) pada detik pengambilan
foto — kalau selisihnya besar, itu sinyal ada kesalahan **fisik** (papan
tidak rata, gnomon tidak tegak sempurna, tinggi gnomon terukur salah, atau
waktu tidak sinkron), bukan sekadar derau pengukuran.

**Status novelty (hasil penelusuran 2 Agustus 2026):** basis gnomonik-nya
(altitude dari rasio tinggi/panjang bayangan) itu sangat klasik — dipakai
sejak zaman kuno untuk menentukan lintang. Konsep membandingkan
altitude+azimuth turunan-bayangan terhadap sumber independen sebagai
pengecekan konsistensi juga sudah ada, tapi dipakai di ranah **forensik
digital** (memverifikasi klaim waktu/lokasi sebuah foto lewat kecocokan
posisi Matahari dari bayangannya). Belum ditemukan penerapannya sebagai
**fitur jaminan mutu real-time** pada instrumen pengukuran arah Kiblat —
kandidat kontribusi kecil tapi jujur dan mudah diverifikasi.

**Usulan langkah teknis:**
1. Setelah garis & panjang bayangan terukur dari bidang papan/tanah (§3).
2. Hitung `altitude_terukur = arctan(tinggi_gnomon / panjang_bayangan)`.
3. Bandingkan dengan `altitude_efemeris` pada detik pengambilan foto.
4. Selisih di luar ambang (mis. >0,5°, sesuai galat gnomonik dasar) →
   peringatan otomatis: "Validasi altitude gagal — periksa ketegakan gnomon /
   kerataan papan / tinggi gnomon yang diinput / sinkronisasi waktu."
5. Selisih dalam ambang → tampilkan sebagai indikator kepercayaan tambahan
   ("✓ Konsistensi altitude terverifikasi") mendampingi hasil azimuth.

**Manfaat:** validasi ini **gratis** — dari foto yang sama yang sudah diambil
untuk azimuth, tanpa alat atau langkah tambahan di lapangan. Cocok jadi fitur
QA otomatis di aplikasi (Tahap 3) sekaligus kontrol mutu data saat
pengumpulan data theodolit (Tahap 2) — dua manfaat sekaligus dari satu
perhitungan tambahan yang murah.

**Pustaka terkait:**
- Gnomonik dasar altitude dari rasio gnomon/bayangan: mis. *Determining Your
  Latitude with a Gnomon* (catatan kuliah, Kevin Krisciunas, Texas A&M).
- Cross-check altitude+azimuth bayangan vs sumber independen: literatur
  forensik verifikasi foto, mis. *Validating the Contextual Information of
  Outdoor Images for Photo Misuse Detection* (arXiv:1811.08951).

---

## 8. Rencana bertahap

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
- Penyempurnaan opsional setelah pipeline dasar jalan **dan** tervalidasi di
  Tahap 2: normalisasi latar berpola (§7.1) dan *burst averaging* dengan
  constraint efemeris (§7.2) — lihat §7 untuk gagasan, status novelty, dan
  kaitan §7.2 dengan analemma. Jangan dikerjakan sebelum Tahap 1–2 selesai.

### Tahap 2 — Validasi lapangan
- Bandingkan CV vs busur manual vs **theodolit** (acuan) pada ≥30 pengukuran,
  beragam altitude (15°–60°), beragam kamera HP dan kondisi cahaya.
- Analisis: bias, simpangan baku, dekomposisi sumber galat, Bland–Altman.
- Ini bagian yang mengubah proyek rekayasa menjadi **penelitian**.
- Aktifkan validasi silang altitude dari panjang bayangan (§7.3) SELAMA
  pengumpulan data ini — dipakai sebagai kontrol mutu tiap sesi pengukuran,
  bukan ditunda ke tahap berikutnya.

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

## 9. Risiko dan batasan (jujur)

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
