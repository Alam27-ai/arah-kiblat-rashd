# Panduan Upload ke GitHub & Menerbitkan Online

Aplikasi ini bisa online **gratis** tanpa server sendiri. Alur: unggah ke GitHub →
deploy ke Streamlit Community Cloud (atau Hugging Face Spaces) → bagikan tautannya.

---

## Bagian 1 — Unggah ke GitHub

### Opsi A — Lewat Git Bash (disarankan; bisa menyertakan efemeris 32 MB)

Unggah web GitHub dibatasi **25 MB per berkas**, sehingga `de440s.bsp` (32 MB)
**tidak bisa** lewat drag-drop web. Gunakan Git Bash (`git push` batas 100 MB):

1. Buat repo kosong di https://github.com → **New repository**
   (*Public*; **jangan** centang Add README / .gitignore / license).
2. Buka **Git Bash** di folder aplikasi lalu jalankan:

```bash
cd "/d/Programming/Arah Kiblat Easy"     # sesuaikan path folder-mu
git init
git add .                                 # .gitignore sudah menyertakan de440s.bsp
git commit -m "Aplikasi Arah Kiblat Easy"
git branch -M main
git remote add origin https://github.com/USERNAME/arah-kiblat-easy.git
git push -u origin main
```

> `.gitignore` sudah dikonfigurasi menyertakan `de440s.bsp` dan mengecualikan
> `de440.bsp`/`de441.bsp` penuh. Jadi `git add .` ikut meng-commit `de440s.bsp`.

### Opsi B — Lewat web (tanpa efemeris; diunduh otomatis)

Bila tak ingin memasang Git: buat repo, klik **uploading an existing file**, seret
semua berkas **kecuali `de440s.bsp`** (karena batas 25 MB), lalu **Commit changes**.
Aplikasi akan mengunduh efemeris otomatis saat pertama dijalankan di server.

---

## Bagian 2 — Deploy ke Streamlit Community Cloud (disarankan)

1. Buka https://share.streamlit.io dan **Sign in with GitHub**.
2. Klik **Create app** → **Deploy a public app from GitHub**.
3. Isi:
   - *Repository*: `USERNAME/arah-kiblat-easy`
   - *Branch*: `main`
   - *Main file path*: `app.py`
4. Klik **Deploy**. Tunggu beberapa menit (memasang dependensi & mengunduh efemeris).
5. Kamu mendapat URL publik, mis. `https://USERNAME-arah-kiblat-easy.streamlit.app`.
   Bagikan tautan ini ke mahasiswa — cukup dibuka di browser, tanpa instal apa pun.

Untuk memperbarui aplikasi: cukup unggah/commit perubahan ke GitHub; Streamlit
Cloud otomatis men-deploy ulang.

---

## Alternatif — Hugging Face Spaces

1. Buat akun di https://huggingface.co → **New Space**.
2. *Space SDK*: **Streamlit**; *Visibility*: Public.
3. Unggah berkas yang sama (bisa lewat antarmuka web). Space otomatis membangun
   dan memberi URL publik.

---

## Catatan Teknis

- **Efemeris `de440s.bsp`**: sudah disertakan di repo, jadi aplikasi online tak
  perlu mengunduh. Bila berkas terhapus, `ensure_ephemeris()` mengunduhnya sekali
  (butuh internet di server) dan menampilkan pesan ramah bila unduhan gagal.
- **Acuan waktu BMKG**: banyak hosting cloud memblokir NTP (UDP 123), sehingga
  aplikasi otomatis memakai **cadangan HTTP** lalu **jam perangkat**. Ini normal
  dan sudah ditangani.
- **Secrets**: aplikasi tidak memerlukan API key atau kredensial apa pun.
- **Dependensi**: seluruhnya tercantum di `requirements.txt`.
