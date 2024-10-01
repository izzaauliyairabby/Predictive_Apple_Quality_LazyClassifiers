# Laporan Proyek Machine Learning - Izza Auliyai Rabby

Dataset ini berisi informasi penting untuk membangun model prediksi kualitas apel berdasarkan faktor visual dan sensorik. Data berasal dari sebuah perusahaan pertanian di Amerika Serikat dan telah diunggah secara publik di platform **Kaggle** dengan judul "_Apple Quality_". Melalui eksplorasi dataset ini, kita dapat mengembangkan model machine learning yang membantu petani dan distributor meningkatkan nilai jual dan kualitas apel mereka.

## Informasi Penting Mengenai Dataset:
- **Jenis Data**: Primer
- **Asal Data**: Perusahaan pertanian di Amerika Serikat

  Tabel 1. EDA Deskripsi Variabel

Dilihat dari Tabel 1. EDA Deskripsi Variabel dataset ini telah di bersihkan dan normalisasi terlebih dahulu oleh pembuat, sehingga mudah digunakan dan ramah bagi pemula.

- Dataset berupa CSV (Comma-Seperated Values).
- Dataset memiliki 4001 sample dengan 9 fitur.
- Dataset memiliki 7 fitur bertipe float64 dan 2 fitur bertipe object.
- Terdapat 1 missing value dalam dataset.
- Variable - variable pada dataset
- A_id : Identifikasi unik untuk setiap buah.
- Size : Ukuran buah.
- Weight : Berat buah.
- Sweetness : Tingkat kemanisan buah.
- Crunchiness : Tekstur yang menunjukkan kerenyahan buah.
- Juiciness : Tingkat kesegaran buah.
- Ripeness : Tahap kematangan buah.
- Acidity : Tingkat keasaman buah.
- Quality : Kualitas buah secara keseluruhan, baik atau buruk.
- 
Dari ke 9 fitur dapat dilihat bahwa fitur A_id tidak mempengaruhi kualitas buah hingga akan di hapus.

Dataset ini mencakup data sensorik (seperti rasa, kerenyahan, dan kadar air) dan data visual (seperti ukuran dan warna) yang dapat digunakan untuk memprediksi kualitas apel.

### Tabel Deskriptif

| Index | Size | Weight | Sweetness | Crunchiness | Juiciness | Ripeness |
|-------|------|--------|-----------|-------------|-----------|----------|
| Count | 4000 | 4000   | 4000      | 4000        | 4000      | 4000     |
| Mean  | -0.50 | -0.99  | -0.47     | 0.98        | 0.51      | 0.50     |
| Std   | 1.93  | 1.60   | 1.94      | 1.40        | 1.93      | 1.87     |
| Min   | -7.15 | -7.14  | -6.89     | -6.06       | -5.96     | -5.86    |
| 25%   | -1.82 | -2.01  | -1.74     | 0.06        | -0.80     | -0.77    |
| 50%   | -0.51 | -0.98  | -0.50     | 0.99        | 0.53      | 0.50     |
| 75%   | 0.81  | 0.03   | 0.80      | 1.89        | 1.84      | 1.77     |
| Max   | 6.41  | 5.79   | 6.37      | 7.62        | 7.36      | 7.24     |

### Exploratory Data Analysis (EDA)
#### 1. **Univariate Analysis**
Data telah dinormalisasi menggunakan metode _z-score_, di mana nilai rata-rata dari setiap fitur dikurangi, kemudian hasilnya dibagi dengan standar deviasi masing-masing. Misalnya, variabel **"Size"** memiliki mean -0.51, menunjukkan bahwa data telah distandarisasi dengan nilai mean mendekati 0 dan standar deviasi 1.

#### 2. **Multivariate Analysis**
Menggunakan _pairplot_, ditemukan pola korelasi menarik, seperti antara **"Size"** dan **"Sweetness"** yang memiliki korelasi negatif. Ini menunjukkan bahwa semakin kecil ukuran apel, semakin manis rasanya.

#### 3. **Matriks Korelasi**
Pada matriks korelasi, hubungan antara berbagai fitur dapat dilihat dengan lebih jelas. Sebagai contoh, **"Juiciness"** memiliki korelasi yang cukup tinggi dengan **"Acidity"** sebesar 0.24, menunjukkan hubungan antara kedua variabel tersebut.

### Data Preparation
Pada tahap ini, berbagai kegiatan dilakukan untuk mempersiapkan data yang bersih dan terstruktur agar siap digunakan dalam proses pemodelan. Beberapa langkah yang dilakukan:
1. **Data Gathering**: Mengimpor data menggunakan Pandas.
2. **Data Assessing**: Memeriksa _duplicate data_, _missing values_, dan _outliers_.
3. **Data Cleaning**: Mengubah tipe kolom, melakukan _train-test split_, dan melakukan normalisasi data agar nilai dari setiap fitur berada dalam skala yang seragam.

### Model Machine Learning yang Digunakan
Untuk memprediksi kualitas apel, beberapa model machine learning digunakan dan dievaluasi untuk mendapatkan model terbaik. Model yang diuji antara lain:

1. **K-Nearest Neighbors (KNN)**: Mengklasifikasikan data berdasarkan tetangga terdekatnya.
2. **Random Forest**: Algoritma ensemble yang menggabungkan banyak pohon keputusan.
3. **Support Vector Machine (SVM)**: Mencari hyperplane terbaik untuk mengklasifikasikan data.
4. **Naive Bayes**: Model probabilistik berdasarkan Teorema Bayes.
5. **Extra Trees Classifier**: Menggunakan banyak pohon keputusan yang tidak dipangkas.

### Evaluasi Model
Metrik utama yang digunakan untuk mengevaluasi model adalah **akurasi**. Akurasi dihitung dengan rumus berikut:

![akurasi](https://github.com/user-attachments/assets/8601ce1f-5c0e-4ab5-b7bb-78560b972bcb)

#### Hasil Perbandingan Model
Berdasarkan hasil evaluasi, model **Random Forest** dan **Extra Trees Classifier** menunjukkan akurasi yang tertinggi dalam memprediksi kualitas apel. Keduanya unggul karena kemampuan menangani kompleksitas data dan menghasilkan prediksi yang lebih akurat melalui penggabungan banyak pohon keputusan.

| Model               | Akurasi   |
|---------------------|-----------|
| K-Nearest Neighbors  | 82%       |
| Random Forest        | 90%       |
| Support Vector Machine | 84%     |
| Naive Bayes          | 78%       |
| Extra Trees Classifier | 92%     |

![plot evaluasi](https://github.com/user-attachments/assets/877dca44-bab7-4c45-a355-6a1554bed67e)


### Kesimpulan dan Aplikasi Praktis
Berdasarkan hasil evaluasi, model **Random Forest** dan **Extra Trees Classifier** dapat digunakan sebagai model terbaik untuk memprediksi kualitas apel. Model ini dapat membantu petani dan distributor dengan cara:

- **Petani**: Dapat meningkatkan hasil panen dengan memilih varietas apel yang lebih baik berdasarkan prediksi kualitas.
- **Distributor**: Dapat meningkatkan efisiensi rantai pasokan dengan memprioritaskan distribusi apel yang berkualitas lebih tinggi, serta menyesuaikan harga berdasarkan prediksi kualitas.

Dengan model prediktif yang akurat, proses pengambilan keputusan dapat lebih didasarkan pada data yang terukur dan dapat diandalkan.
