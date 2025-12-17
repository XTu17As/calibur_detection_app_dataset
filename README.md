# Calibur Detection App Dataset dan Kode Training

Repositori ini berfungsi sebagai arsip penyimpanan data dan kode sumber yang digunakan dalam pengembangan model deteksi objek untuk penelitian skripsi. Repositori ini mencakup dataset utama, dataset eksperimental untuk pengujian variasi sudut pandang, serta algoritma pelatihan model.

## 🗂️ Struktur dan Deskripsi File
### Lihat file .txt
Berkas-berkas dalam repositori ini dikategorikan berdasarkan fungsinya dalam pengembangan model:

### 1. Dataset Utama
* **Nama Berkas**: `Skripsi_Aug_384_from1080_alt4.rar`
* **Deskripsi**:
    Arsip ini berisi kumpulan data primer yang digunakan untuk melatih model deteksi utama. Data di dalamnya telah melalui proses pra-pemrosesan dan augmentasi data yang komprehensif untuk memastikan generalisasi model yang optimal pada lingkungan produksi. Dataset ini merupakan acuan standar (baseline) dalam penelitian ini.

### 2. Dataset Eksperimental (Frontal)
* **Nama Berkas**: `Skripsi_SplitThenAug_384_Threads_frontal_v5.rar`
* **Deskripsi**:
    Arsip ini berisi variasi dataset yang dikhususkan untuk eksperimen model dengan sudut pandang terbatas (frontal). Dataset ini dipisahkan dan diproses secara spesifik untuk menguji hipotesis kinerja model pada kondisi pengambilan gambar satu arah dan digunakan sebagai pembanding terhadap model utama.

### 3. Kode Pelatihan (*training_code.py*)
* **Deskripsi**:
    Direktori ini memuat seluruh kode algoritma yang diperlukan untuk siklus hidup pengembangan model. Cakupan kode meliputi:
    * Skrip untuk memuat dan memproses data (*data loading & preprocessing*).
    * Definisi arsitektur model.
    * Algoritma pelatihan (*training loop*) dan validasi.
    * Konfigurasi parameter untuk reproduktabilitas hasil eksperimen.

## 💻 Cara Penggunaan

1.  **Ekstraksi Data**: Unduh dan ekstrak berkas `.rar` sesuai dengan model yang ingin dikembangkan atau diuji (Utama atau Eksperimental).
2.  **Persiapan Lingkungan**: Pastikan seluruh dependensi *library* yang tertera pada kode pelatihan telah terinstal.
3.  **Pelatihan Model**: Gunakan skrip di dalam `training_code` dan arahkan direktori data ke hasil ekstraksi dataset yang telah diunduh.

## 📝 Catatan Tambahan

Seluruh data yang tersedia dalam repositori ini ditujukan semata-mata untuk kepentingan penelitian dan pengembangan sistem deteksi terkait skripsi ini. Penggunaan di luar konteks tersebut memerlukan penyesuaian pada parameter kode pelatihan.
