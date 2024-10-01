# Laporan Proyek Machine Learning - Izza Auliyai Rabby
## Domain Proyek
Domain yang dipilih untuk proyek *machine learning* ini adalah **Pertanian**, dengan judul **Predictive Analytics: Kualitas Apel**  

### Latar Belakang
Indonesia merupakan salah satu negara penghasil apel terbesar di Asia Tenggara, dengan produksi mencapai 1,2 juta ton per tahun. Apel menjadi komoditas penting bagi petani dan berkontribusi signifikan pada perekonomian nasional.[[1](https://dataindonesia.id/agribisnis-kehutanan/detail/produksi-apel-indonesia-sebanyak-509544-ton-pada-2021)] Salah satu tantangan utama dalam industri apel adalah menjaga kualitas produk. Faktor-faktor seperti ukuran yang kecil, tingkat kematangan, dan kerenyahan dapat mempengaruhi kualitas apel. Penurunan kualitas apel dapat menyebabkan kerugian ekonomi bagi petani dan distributor.[[2](https://hostjournals.com/bulletincsr/article/view/251)] Penerapan _predictive analytics_ dalam industri apel dapat memberikan manfaat bagi petani, distributor, dan konsumen. Petani dapat meningkatkan keuntungan dengan memperbaiki kualitas dan hasil panen apel. Distributor dapat mengurangi kerugian dan meningkatkan efisiensi rantai pasokan. Konsumen akan mendapatkan apel dengan kualitas lebih baik dan harga yang lebih stabil.[[3](https://doi.org/10.47065/bulletincsr.v3i3.251)]

## Business Understanding
Pengembangan model prediksi kualitas apel memiliki potensi untuk memberikan manfaat signifikan bagi berbagai pihak, seperti petani dan distributor. Model prediktif ini dapat membantu meningkatkan kualitas panen apel, meningkatkan nilai jual, serta memperkuat kepercayaan konsumen. Sebagai contoh, prediksi kualitas apel yang akurat dapat membantu petani dalam proses pemilahan serta menentukan harga jual buah secara lebih tepat di masa mendatang.

### Problem Statements
Berdasarkan latar belakang di atas, berikut adalah masalah-masalah yang dapat diselesaikan oleh proyek ini:
- Bagaimana cara membangun model machine learning yang mampu memprediksi kualitas apel berdasarkan data visual dan sensorik?
- Model machine learning seperti apa yang memiliki akurasi terbaik dalam memprediksi kualitas apel?
- Bagaimana model ini dapat membantu petani dan distributor dalam meningkatkan kualitas serta nilai jual apel?

### Goals
Tujuan dari proyek ini meliputi:
- Mengembangkan model machine learning yang dapat memprediksi kualitas apel menggunakan data visual dan sensorik.
- Membandingkan berbagai algoritma untuk menemukan model dengan akurasi terbaik dalam memprediksi kualitas apel.
- Mengembangkan aplikasi yang mudah digunakan oleh petani dan distributor, yang memungkinkan mereka memanfaatkan model machine learning untuk memprediksi kualitas apel.

### Solution Statements
Menganalisis data dengan melakukan univariate analysis dan multivariate analysis. Memahami data juga dapat dilakukan dengan visualisasi. Memahami data dapat membantu untuk mengetahui kolerasi matrix antar fitur dan mendeteksi outlier.
Melakukan proses data cleaning dan normalisai data agar mendapat prediksi yang baik.
Membuat beberapa variasi model untuk mendapatkan model yang paling baik dari beberapa model yang telah dibuat untuk prediksi kualitas apel. Diantaranya adalah menggunakan:
K-Nearest Neighbor (KNN) adalah algoritma sederhana yang mengklasifikasikan data atau kasus baru berdasarkan ukuran kesamaan. Hal ini sebagian besar digunakan untuk mengklasifikasikan titik data berdasarkan tetangga terdekatnya sebagai acuan.[4]
Random Forest adalah algoritma machine learning yang kuat yang dapat digunakan untuk berbagai tugas termasuk regresi dan klasifikasi. Ini adalah metode ensemble, yang berarti bahwa model random forest terdiri dari banyak decision tree kecil, yang disebut estimator, yang masing-masing menghasilkan prediksi mereka sendiri. Random forest menggabungkan prediksi estimator untuk menghasilkan prediksi yang lebih akurat .[5]
Support Vector Machine (SVM) adalah algoritma yang digunakan untuk menemukan hyperplane dalam ruang N-dimensi (N - jumlah fitur) yang secara jelas mengklasifikasikan titik data. SVM dapat digunakan untuk menyelesaikan permasalahan klasifikasi, regresi, dan pendeteksian outlier.[6]
Naive Bayes adalah model machine learning probabilistik yang digunakan untuk tugas klasifikasi. Inti dari classifier ini didasarkan pada teorema Bayes.[7]
Extra trees classifier adalah sejumlah besar pohon keputusan yang belum dipangkas dari kumpulan data pelatihan. Prediksi dibuat dengan merata-ratakan prediksi pohon keputusan dalam kasus regresi atau menggunakan suara terbanyak dalam kasus klasifikasi.[8]

Melalui pendekatan ini, model dengan akurasi terbaik akan dipilih untuk memprediksi kualitas apel dengan lebih efektif.

## Data Understanding
**Informasi Datasets**

| Jenis        | Keterangan                                                                                  |
|--------------|---------------------------------------------------------------------------------------------|
| **Title**    | _Apple Quality_                                                                             |
| **Source**   | [Kaggle](https://www.kaggle.com/datasets/nelgiriyewithana/apple-quality/data)                |
| **Maintainer**| [Nidula Elgiriyewithana ⚡](https://www.kaggle.com/nelgiriyewithana)                        |
| **License**  | Other (specified in description)                                                            |
| **Visibility**| Publik                                                                                      |
| **Tags**     | _Computer Science, Education, Food, Data Visualization, Classification, Exploratory Data Analysis_ |
| **Usability**| 10.00                                                                                        |

Dataset ini berisi informasi yang diperlukan untuk memprediksi kualitas apel berdasarkan berbagai faktor yang relevan. Data tersebut bersumber dari sebuah perusahaan pertanian di Amerika dan telah disediakan secara publik di Kaggle dengan judul "_Apple Quality_". Dataset ini memungkinkan eksplorasi mendalam terkait prediksi kualitas apel menggunakan data visual dan sensorik, yang akan membantu dalam pengembangan model machine learning untuk meningkatkan nilai jual dan kualitas apel.

Berikut informasi penting mengenai dataset: 

- **Jenis Data**: Primer
- **Asal Data**: Perusahaan pertanian Amerika

Dengan memanfaatkan dataset ini, analisis eksploratif serta pengembangan model prediksi kualitas apel dapat dilakukan secara optimal.

|index|Size|Weight|Sweetness|Crunchiness|Juiciness|Ripeness|
|---|---|---|---|---|---|---|
|count|4000\.0|4000\.0|4000\.0|4000\.0|4000\.0|4000\.0|
|mean|-0\.50301462982675|-0\.9895465445945|-0\.47047851978824995|0\.9854779038585|0\.5121179684932501|0\.4982774280305|
|std|1\.928058688854979|1\.6025072141517547|1\.943440658920452|1\.402757204211963|1\.9302856730942946|1\.8744267757033417|
|min|-7\.151703059|-7\.149847675|-6\.894485494|-6\.055057805|-5\.961897048|-5\.864598918|
|25%|-1\.816764527|-2\.01177029275|-1\.7384250625|0\.06276439525000001|-0\.80128581525|-0\.7716768665|
|50%|-0\.5137025125000001|-0\.9847364865|-0\.5047584635|0\.9982494390000001|0\.5342186584999999|0\.5034447135|
|75%|0\.8055264495000001|0\.03097644|0\.8019219209999999|1\.8942342170000002|1\.8359763875|1\.76621164075|
|max|6\.406366899|5\.79071359|6\.374915513|7\.619851801|7\.364402864|7\.237836684|

### Tabel 1. EDA Deskripsi Variabel & Univariate Analysis
Dalam proses **Exploratory Data Analysis (EDA)**, kita mengamati bahwa data telah dinormalisasi menggunakan metode _z-score normalization_. Normalisasi ini dilakukan dengan cara:

- Mengurangi rata-rata (mean) dari setiap data point.
- Membagi hasil pengurangan tersebut dengan standar deviasi data.

Misalnya, variabel **"Size"** memiliki mean -0.51, tetapi standar deviasi tidak diketahui. Namun, berdasarkan nilai minimum (-2) dan maksimum (2), dapat diasumsikan bahwa data telah distandarisasi sehingga memiliki mean 0 dan standar deviasi 1. Hal ini juga berlaku untuk variabel numerik lainnya seperti **"Weight", "Sweetness", "Crunchiness", "Juiciness", "Ripeness"**, dan **"Acidity"**.

### EDA - Multivariate Analysis

- **Gambar 2a. Analisis Multivariat**:
  Menggunakan _pairplot_ dari _library Seaborn_, terlihat bahwa pola pasangan antar variabel tampak acak. Salah satu pola yang menarik adalah antara **"Size"** dan **"Sweetness"**, di mana terlihat korelasi negatif: semakin kecil ukuran apel, semakin manis rasanya.

- **Gambar 2b. Matriks Korelasi**:
  Pada matriks korelasi, kita bisa melihat hubungan antar fitur. Sebagai contoh, **"Juiciness"** memiliki korelasi yang cukup tinggi dengan **"Acidity"** sebesar `0.24`, yang menunjukkan adanya hubungan antara kedua variabel tersebut.

### Data Preparation
Pada tahap **Data Preparation**, beberapa langkah penting dilakukan untuk mempersiapkan data agar siap digunakan dalam model prediksi:

1. **Data Gathering**: Data dikumpulkan dan diimpor dengan benar menggunakan _dataframe_ Pandas agar mudah diakses dan dianalisis.
   
2. **Data Assessing**: 
   - **Duplicate data**: Memeriksa adanya duplikasi data.
   - **Missing value**: Mengidentifikasi data yang hilang.
   - **Outlier**: Memeriksa adanya outlier, yaitu data yang menyimpang secara signifikan dari pola umum.

3. **Data Cleaning**: 
   - **Converting Column Type**: Mengubah tipe data kolom yang tidak sesuai.
   - **Train-Test Split**: Membagi data menjadi **data latih** dan **data uji** untuk memastikan performa model yang valid.
   - **Normalization**: Mentransformasi data sehingga semua fitur memiliki skala yang sebanding, menjaga konsistensi dalam proses pemodelan.

  |index|Size|Weight|Sweetness|Crunchiness|Juiciness|Ripeness|Acidity|Quality|
|---|---|---|---|---|---|---|---|---|
|4000|NaN|NaN|NaN|NaN|NaN|NaN|Created\_by\_Nidula\_Elgiriyewithana|NaN|

## Tabel 2. Melihat Data Missing Value

Pada proses analisis, kita juga melakukan pengecekan terhadap data _missing value_. Ketika bekerja dengan dataset kompleks, adanya data yang hilang dapat mempengaruhi kinerja model, terutama jika data tersebut signifikan dalam menentukan hasil prediksi. Oleh karena itu, penting untuk mengidentifikasi dan menangani _missing values_ sebelum melanjutkan ke tahap pemodelan.

---

### Extra Trees Classifier

Algoritma _Extra Trees Classifier_ adalah salah satu metode _ensemble learning_ yang digunakan untuk mengklasifikasikan data. Algoritma ini mirip dengan _Random Forest_, tetapi memiliki beberapa perbedaan utama, yaitu:
- **Random Splitting**: Dalam _Extra Trees_, pemisahan dilakukan secara acak, bukan berdasarkan pencarian pemisahan terbaik seperti pada _Random Forest_.
- **No Bagging**: Tidak dilakukan pemilihan sampel acak (_bagging_), sehingga semua data digunakan untuk membangun pohon keputusan.

#### Keuntungan Extra Trees Classifier:
- Lebih tahan terhadap **overfitting** dibandingkan dengan _Random Forest_, terutama pada dataset berdimensi tinggi.
- **Mudah diimplementasikan** dan digunakan.
- Kinerja yang baik pada berbagai masalah klasifikasi.

#### Kerugian Extra Trees Classifier:
- Pada beberapa dataset, hasil klasifikasi cenderung **kurang akurat** dibandingkan dengan _Random Forest_.
- Membutuhkan **waktu komputasi yang tinggi** untuk pelatihan, terutama pada dataset yang besar.

#### Parameter yang Digunakan:
- `n_estimators`: Jumlah pohon keputusan yang akan dibuat dalam _ensemble_.
- `random_state`: Pengaturan untuk pengambilan sampel acak.
- `max_depth`: Kedalaman maksimum dari setiap pohon keputusan.
- `n_jobs`: Menentukan jumlah core CPU yang digunakan untuk mempercepat pelatihan.

---

## Evaluation

Dalam tahap evaluasi, metrik utama yang digunakan adalah **accuracy**. Akurasi mengukur sejauh mana prediksi yang dilakukan oleh model sesuai dengan nilai aktual dalam data uji. Rumus untuk menghitung akurasi adalah:

\[
\text{Accuracy} = \frac{\text{TP + TN}}{\text{TN + TP + FN + FP}} \times 100\%
\]

#### Penjelasan:
- **TP (True Positive)**: Jumlah data positif yang diklasifikasikan dengan benar sebagai positif.
- **TN (True Negative)**: Jumlah data negatif yang diklasifikasikan dengan benar sebagai negatif.
- **FP (False Positive)**: Jumlah data negatif yang salah diklasifikasikan sebagai positif.
- **FN (False Negative)**: Jumlah data positif yang salah diklasifikasikan sebagai negatif.

Rumus ini menggambarkan rasio antara prediksi yang benar (TP dan TN) dengan jumlah total prediksi, dikalikan dengan 100% untuk mendapatkan persentase akurasi.

### Hasil Evaluasi Akurasi:
Berikut adalah hasil akurasi dari lima model yang dilatih:

| Model                   | Akurasi (%) |
|--------------------------|-------------|
| K-Nearest Neighbor (KNN)  |  0.9010554089709762    |
| Random Forest             |0.8720316622691293       |
| Support Vector Machine    |0.8825857519788918       |
| Naive Bayes               |0.5092348284960422       |
| Extra Trees Classifier    |0.8601583113456465       |

Catatan: Hasil akurasi setiap model perlu diukur secara lebih mendetail berdasarkan data dan evaluasi lebih lanjut.

![plot evaluasi](https://github.com/user-attachments/assets/950660ca-6bed-475e-99b2-8394ceb8225c)


Deskripsi Plot:
Plot ini menampilkan perbandingan akurasi dari berbagai model machine learning yang telah dievaluasi pada dataset apel. 
Sumbu X menunjukkan model yang digunakan, dan sumbu Y menunjukkan skor akurasi yang dicapai oleh masing-masing model.
Berdasarkan plot ini, kita dapat dengan mudah membandingkan kinerja setiap model dan mengidentifikasi model mana yang memiliki akurasi terbaik dalam memprediksi kualitas apel.
Model dengan akurasi tertinggi dapat digunakan sebagai model terbaik untuk memprediksi kualitas apel pada dataset yang sama atau pada dataset baru yang serupa.

Analisis dan Interpretasi:
Berdasarkan perbandingan model, model Random Forest dan Extra trees classifier menunjukkan akurasi yang tinggi dalam memprediksi kualitas apel.
Ini mungkin karena kemampuan model ini untuk menangani kompleksitas data dan membuat keputusan berdasarkan banyak fitur.
Model lainnya seperti KNN, SVM, dan Naive Bayes juga menunjukkan akurasi yang cukup baik, tetapi mungkin kurang optimal dibandingkan Random Forest dan Extra trees classifier dalam dataset ini.

## Referensi
1. Sarnita Sadya.(2022). Produksi Apel Indonesia Sebanyak 509.544 Ton pada 2021.

2. Lomo, Christine P., et al. "Daya Terima Panelist terhadap Kualitas Cider Apel dalam Meningkatkan Nilai Gizi Pangan sebagai Imunitas Tubuh di Pandemi Covid-19." Agrista: Jurnal Ilmiah Mahasiswa Agribisnis UNS, vol. 4, no. 1, 2020, pp. 550-556
3. Afriansyah, M., Saputra, J., Sa’adati, Y., & Valian Yoga Pudya Ardhana. (2023). Optimasi Algoritma Nai?ve Bayes Untuk Klasifikasi Buah Apel Berdasarkan Fitur Warna RGB. Bulletin of Computer Science Research, 3(3), 242-249.
4. Subramanian, D. (2019). A Simple Introduction to K-Nearest Neighbors Algorithm. Towards Data Science. https://towardsdatascience.com/a-simple-introduction-to-k-nearest-neighbors-algorithm-b3519ed98e
5. Wood, T. -.What is a Random Forest?. DeepAI. https://deepai.org/machine-learning-glossary-and-terms/random-forest
6. Gandhi, R. (2018). Support Vector Machine — Introduction to Machine Learning Algorithms: SVM model from scratch. Towards Data Science. https://towardsdatascience.com/support-vector-machine-introduction-to-machine-learning-algorithms-934a444fca47
7. Gandhi, R. (2018). Naive Bayes Classifier. Towards Data Science. https://towardsdatascience.com/naive-bayes-classifier-81d512f50a7c
8. Jason Brownlee. (2021). How to Develop an Extra Trees Ensemble with Python. https://machinelearningmastery.com/extra-trees-ensemble-with-python/
