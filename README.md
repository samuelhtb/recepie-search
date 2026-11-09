# Sistem Temu Kembali Informasi Resep Masakan Indonesia — BM25

Implementasi **Pipeline Rancangan Project TKI – Kelompok 3** dan artikel **Sistem Temu Kembali Informasi Resep Masakan Indonesia Berbasis BM25**, dengan dua halaman Streamlit:

1. **🔎 Tes Pencarian Resep** — mencari Top-5 berdasarkan judul, bahan, langkah, atau kombinasinya.
2. **📊 Evaluasi & Grafik** — menyiapkan Q1–Q7 dari data eval, menilai kandidat secara manual oleh dua orang, menghitung P@5/MRR@5/nDCG@5/waktu retrieval, dan menampilkan tiga grafik beserta penjelasan.

Tidak menggunakan neural network, embedding, reranking, rekomendasi resep baru, atau label relevansi otomatis.

## Persiapan dan menjalankan aplikasi

Buka terminal **di dalam folder `recipe-search-ir`**. Disarankan Python 3.10+.

```bash
python -m venv .venv
```

Aktifkan virtual environment:

- Windows PowerShell: `.\.venv\Scripts\Activate.ps1`
- Windows CMD: `.venv\Scripts\activate.bat`
- macOS/Linux: `source .venv/bin/activate`

Kemudian jalankan **berurutan**:

```bash
python -m pip install -r requirements.txt
python -m src.data_loader
python -m src.indexing
python -m streamlit run app.py
```

Browser biasanya membuka `http://localhost:8501`. Gunakan menu navigasi **di sebelah kiri** untuk berpindah antara dua halaman. Jika ingin menjalankan aplikasi kembali, cukup `python -m streamlit run app.py`; indeks tidak perlu dibangun ulang selama corpus dan preprocessing tidak berubah.

**Dataset wajib berada di:**

```text
data/raw/train.parquet
data/raw/eval.parquet
```

Versi ZIP *lengkap* sudah menyediakan kedua file. Versi ZIP *kode saja* tidak menyertakan dataset: salin kedua file secara manual ke folder tersebut.

> `python -m src.indexing` harus dijalankan **satu kali sebelum membuka halaman pencarian/evaluasi**. Proses ini memakan waktu karena corpus relatif besar (sekitar 66 ribu resep sebelum pembersihan). File `artifacts/bm25.pkl`, `artifacts/recipes.pkl`, dan `artifacts/manifest.json` dibuat otomatis.

## Halaman 1 — Tes Pencarian Resep

Masukkan minimal satu dari tiga kolom:

- **Judul**: contoh `Ayam Goreng Bawang Putih`.
- **Bahan**: contoh `ayam, bawang putih, garam`.
- **Langkah**: contoh `goreng ayam hingga matang`.

Klik **Cari Resep**. Hasil menampilkan maksimal **Top-5**, ranking, skor BM25, dan rincian bahan serta langkah lengkap dalam panel yang dapat diperluas.

Kode kombinasi yang ditangani otomatis: Q1 (judul), Q2 (bahan), Q3 (langkah), Q4 (judul+bahan), Q5 (judul+langkah), Q6 (bahan+langkah), Q7 (semuanya). Skor BM25 **bukan** probabilitas, persentase relevansi, atau penilaian kualitas masakan.

## Halaman 2 — Evaluasi & Grafik

Semua tahap utama bisa dilakukan **langsung di Streamlit**, tanpa perlu membuka CSV atau menjalankan perintah evaluasi di terminal.

### Tab 1: Persiapan Pengujian

1. Pilih jumlah resep dasar **10–15** (default **12**) dari `eval.parquet`.
2. Biarkan seed **42** agar hasil pemilihan awal dapat direplikasi.
3. Klik **Buat / Bangun Ulang Bahan Evaluasi**.
4. Program membentuk 7 query per resep dasar (**84 query** jika 12 resep), menjalankan BM25 pada corpus train yang sama, dan mengumpulkan kandidat **Top-10** (hanya yang memiliki skor BM25 > 0).

File yang dihasilkan di folder `data/processed/`:

| File | Kegunaan |
|---|---|
| `queries.csv` | ID query, variasi Q1–Q7, masukan query, waktu retrieval. |
| `retrieval_pool.csv` | Daftar kandidat Top-10 per query beserta judul, bahan, dan langkah. |
| `relevance_judgments.csv` | Nilai dari dua penilai dan label hasil diskusi. |
| `evaluation_setup.json` | Metadata banyaknya query, seed, dan kandidat. |

**Penting:** menjalankan kembali Persiapan dengan izin *timpa* akan **menghapus seluruh anotasi lama**. Sistem meminta centang konfirmasi sebelum menimpa eksperimen lama.

### Tab 2: Penilaian Relevansi (dua orang)

1. Pilih **Penilai 1**. Pilih query, buka rincian resep dalam daftar kandidat, isi setiap kolom nilai **0/1/2**, lalu klik **Simpan Penilaian**. Ulangi untuk seluruh query.
2. Pilih **Penilai 2** dan lakukan hal yang sama secara independen. Kolom penilai lain tidak ditampilkan dalam mode awal.
3. Jika ada perbedaan, pilih **Diskusi / nilai akhir**. Hanya kandidat yang tidak disepakati akan muncul. Diskusikan dan isi nilai final.
4. Periksa indikator progres, jumlah kandidat yang dinilai dan konflik belum terselesaikan. Tab evaluasi hanya bisa menghitung metrik jika semua anotasi selesai.

Kriteria sesuai laporan penelitian:

| Nilai | Tingkat | Pengertian |
|---|---|---|
| **0** | Tidak relevan | Tidak sesuai kebutuhan query. |
| **1** | Cukup relevan | Sesuai dengan sebagian informasi query. |
| **2** | Sangat relevan | Sangat sesuai dengan kebutuhan query. |

Keputusan akhir **tidak dihitung otomatis oleh BM25**. Jika dua penilai sudah sepakat, nilai final boleh dikosongkan; program otomatis memakai kesepakatan. Jika berbeda, hasil diskusi **wajib diisi**. Mengubah nilai salah satu penilai akan mengosongkan keputusan akhir yang lama pada baris terkait agar tidak memakai keputusan usang.

Penilaian disimpan langsung ke `data/processed/relevance_judgments.csv` agar tidak hilang saat Streamlit ditutup. Tersedia tombol **Unduh CSV** untuk pencadangan/kolaborasi. Hindari dua orang menyimpan file yang sama **secara bersamaan** pada satu server, karena implementasi ini masih berbasis file lokal, bukan sistem multi-pengguna berbasis database.

### Tab 3: Hasil & Grafik

Setelah dua penilai menyelesaikan semua kandidat, klik **Hitung / Perbarui Metrik Evaluasi**. Akan muncul empat angka ringkasan, grafik, tabel, dan tombol unduh CSV.

**Empat metrik:**

- **Precision@5**: proporsi Top-5 yang dinilai relevan (1 atau 2), penyebut **tetap 5**.
- **MRR@5**: rata-rata kebalikan posisi hasil relevan pertama di Top-5. Jika tidak ada, nilainya nol.
- **nDCG@5**: kualitas ranking menggunakan relevansi bertingkat 0/1/2 dan `2^rel - 1`, dibandingkan ranking ideal berdasarkan **pool yang telah dinilai**. Jika ideal DCG = 0, nDCG = 0.
- **Retrieval time (ms)**: lama pemrosesan query, perhitungan BM25, dan ranking yang dicatat saat mempersiapkan query.

**Tiga grafik interaktif:**

1. **Perbandingan P@5, MRR@5, dan nDCG@5 Q1–Q7**. Semua metrik tersebut berada pada rentang 0–1, makin besar makin baik sesuai definisi metrik.
2. **Waktu retrieval rata-rata dan median per kombinasi**. Makin kecil makin cepat pada pengujian yang sama. Satuannya *milidetik*, sehingga diplot terpisah dari metrik relevansi.
3. **Heatmap nDCG@5 per resep dasar dan Q1–Q7**. Memperlihatkan apakah keberhasilan retrieval konsisten antarjenis resep, bukan hanya nilai rata-rata.

Di bawah grafik terdapat **interpretasi otomatis yang hanya menggunakan angka hasil penilaian sebenarnya**, batasan metodologi, tabel seluruh kombinasi, serta CSV `metrics_by_query.csv`, `metrics_by_combination.csv`, dan `evaluation_summary.json`.

**Jangan mengartikan skor tinggi sebagai kesimpulan umum semua resep:** pengujian menggunakan sekitar 10–15 resep dasar, dan penilai hanya memeriksa kumpulan kandidat yang ditemukan (maksimal 10 per query), bukan semua dokumen relevan dalam corpus. Program **menyembunyikan grafik lama** ketika file anotasi diubah, sampai metrik dihitung ulang.

## Alternatif terminal untuk evaluasi

Jika tidak ingin menggunakan UI, alur terminal dari versi sebelumnya **tetap tersedia**:

```bash
python -m src.evaluate prepare --base 12 --pool 10 --seed 42
# Isi data/processed/relevance_judgments.csv secara manual (dua penilai)
python -m src.evaluate score
```

Alur ini menghasilkan file CSV/metrik yang sama, sehingga hasilnya juga dapat dibaca di halaman Evaluasi & Grafik, selama anotasi sudah lengkap.

## Struktur kode

```text
recipe-search-ir/
├── app.py                       # st.navigation: dua halaman
├── pages/
│   ├── pencarian.py             # halaman tes pencarian Top-5
│   └── evaluasi.py              # halaman evaluasi + 3 grafik Plotly
├── src/
│   ├── data_loader.py           # audit dan pembersihan corpus
│   ├── preprocess.py            # regex, casefold, tokenisasi
│   ├── indexing.py              # BM25Okapi + simpan/muat indeks
│   ├── query_processing.py      # Q1–Q7
│   ├── retrieval.py             # skor/ranking/waktu
│   ├── evaluate.py              # query pengujian, CSV, metrik
│   ├── evaluation_dashboard.py  # anotasi UI, progres, cek hasil usang
│   └── ui_shared.py             # path proyek dan cache model Streamlit
├── data/
│   ├── raw/train.parquet
│   ├── raw/eval.parquet
│   └── processed/               # file evaluasi (dibuat otomatis)
├── artifacts/                   # indeks BM25 (dibuat otomatis)
├── requirements.txt
└── tests/
```

## Keputusan teknis baseline

- `rank_bm25.BM25Okapi`, parameter `k1=1.5` dan `b=0.75`, **sama pada Q1–Q7**.
- Representasi corpus = `title + ingredients + steps`, tanpa bobot field.
- Preprocessing lowercase (`casefold`) + tokenisasi regex, frekuensi istilah tetap dipertahankan, **tanpa stemming maupun stopword removal**.
- Duplikat penuh (judul+bahan+langkah) dibuang; judul sama dengan komposisi/langkah berbeda tetap boleh ada. ID dokumen mengikuti nomor baris sumber.
- Hasil pencarian berskor > 0, sorting skor menurun dan `doc_id` menaik untuk skor yang sama.
- Eval **tidak** masuk corpus. Resep eval yang identik secara penuh dengan train dikeluarkan dari kandidat query awal. Kemiripan *near-duplicate* tidak disaring otomatis; tim harus meninjaunya sebelum evaluasi akhir.
- Judgement pool Top-10 (bisa kurang jika hasil dengan skor positif terbatas), dua penilai manusia; metrik dibatasi maksimal Top-5.
- Objek indeks disimpan sebagai file `pickle`: **hanya muat file artefak yang dibuat sendiri dan terpercaya**.

## Uji kode

```bash
python -m pytest -q
```

Pengujian unit menguji data loader, query, retrieval, perhitungan metrik, progress anotasi, penyimpanan label, dan pembaruan hasil evaluasi. Pengujian integrasi visual UI serta indexing seluruh 66 ribu resep perlu dilakukan pada komputer yang telah menginstal dependensi di `requirements.txt`.
