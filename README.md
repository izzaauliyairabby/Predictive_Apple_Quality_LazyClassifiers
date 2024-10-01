# Laporan Proyek Machine Learning - Izza Auliyai Rabby
## Domain Proyek
Domain yang dipilih untuk proyek *machine learning* ini adalah **Pertanian**, dengan judul **Predictive Analytics: Kualitas Apel**  

### Latar Belakang
Indonesia merupakan salah satu produsen apel terbesar di Asia Tenggara, dengan total produksi mencapai 1,2 juta ton per tahun. Komoditas apel memiliki peran penting bagi para petani dan berkontribusi signifikan terhadap perekonomian nasional [[1](https://dataindonesia.id/agribisnis-kehutanan/detail/produksi-apel-indonesia-sebanyak-509544-ton-pada-2021)]. Salah satu tantangan utama dalam sektor ini adalah menjaga kualitas produk. Faktor-faktor seperti ukuran kecil, tingkat kematangan, dan kerenyahan dapat memengaruhi kualitas apel, dan penurunan kualitas tersebut dapat menyebabkan kerugian ekonomi bagi petani serta distributor [[2](https://hostjournals.com/bulletincsr/article/view/251)]. Dengan penerapan _predictive analytics_ di industri apel, petani, distributor, dan konsumen dapat memperoleh berbagai manfaat. Petani dapat meningkatkan pendapatan melalui peningkatan kualitas dan hasil panen, sementara distributor dapat mengurangi kerugian serta meningkatkan efisiensi rantai pasokan. Konsumen juga akan diuntungkan dengan mendapatkan apel berkualitas lebih baik dan harga yang lebih stabil [[3](https://doi.org/10.47065/bulletincsr.v3i3.251)].

## Business Understanding

Pengembangan model prediksi kualitas apel berpotensi memberikan manfaat besar bagi berbagai pemangku kepentingan, seperti petani dan distributor. Model prediktif ini dapat membantu dalam meningkatkan kualitas hasil panen, menaikkan nilai jual apel, serta memperkuat kepercayaan konsumen. Sebagai ilustrasi, prediksi kualitas apel yang akurat akan mempermudah petani dalam proses pemilahan dan memungkinkan penentuan harga jual yang lebih tepat di masa depan.

### Problem Statements
Berdasarkan latar belakang yang telah dijelaskan, berikut adalah beberapa masalah yang dapat diselesaikan oleh proyek ini:

Bagaimana cara membangun model machine learning yang mampu memprediksi kualitas apel berdasarkan data visual dan sensorik?
Masalah ini mengarah pada pengembangan model yang menggunakan kombinasi data visual (seperti warna dan tekstur) serta data sensorik (seperti rasa, kerenyahan, dan kadar air) untuk memprediksi kualitas apel.

Model machine learning seperti apa yang memiliki akurasi terbaik dalam memprediksi kualitas apel?
Di sini, fokusnya adalah membandingkan berbagai algoritma machine learning untuk menemukan model yang memberikan akurasi terbaik, dengan mempertimbangkan metrik evaluasi seperti akurasi, presisi, dan recall.

Bagaimana model ini dapat membantu petani dan distributor dalam meningkatkan kualitas serta nilai jual apel?
Masalah ini menyasar pada aplikasi praktis dari model, yaitu bagaimana prediksi kualitas yang lebih akurat dapat mendukung proses pengambilan keputusan bagi petani dalam meningkatkan hasil panen, serta membantu distributor dalam manajemen rantai pasokan dan harga jual.

### Goals
Tujuan dari proyek ini adalah sebagai berikut:

Mengembangkan model machine learning yang dapat memprediksi kualitas apel menggunakan data visual (seperti warna dan tekstur) serta data sensorik (seperti rasa, tingkat kematangan, dan kadar air) untuk membantu petani dan distributor mengoptimalkan hasil panen.
Membandingkan berbagai algoritma machine learning guna menemukan model dengan akurasi terbaik dalam memprediksi kualitas apel. Algoritma yang dibandingkan akan mencakup metode seperti Random Forest, Extra Trees Classifier, Support Vector Machine (SVM), dan algoritma lain yang relevan.
Mengembangkan aplikasi yang mudah digunakan oleh petani dan distributor, yang memanfaatkan model machine learning tersebut untuk memprediksi kualitas apel secara real-time. Aplikasi ini diharapkan mampu memberikan hasil prediksi yang akurat dan membantu pengambilan keputusan terkait panen dan distribusi apel.

### Solution Statements
Untuk mencapai prediksi kualitas apel yang optimal, beberapa langkah penting dalam analisis data dan modelisasi harus dilakukan:

1. **Analisis Univariate dan Multivariate**: 
   - **Univariate analysis** membantu memahami distribusi setiap fitur secara individual. Ini meliputi analisis statistik deskriptif (seperti mean, median, dan standar deviasi) dan visualisasi (seperti histogram dan boxplot) untuk mendeteksi pola, tren, dan outlier.
   - **Multivariate analysis** mengamati hubungan antar beberapa fitur sekaligus. Teknik seperti **korelasi matriks** dapat digunakan untuk melihat hubungan antar variabel. Visualisasi melalui **pair plots** atau **heatmaps** sangat berguna untuk mendeteksi hubungan kuat antara fitur.

2. **Data Cleaning dan Normalisasi**:
   - **Proses cleaning data** meliputi penghapusan atau imputasi missing values, penanganan outlier, dan menangani data yang tidak konsisten. 
   - **Normalisasi data** diperlukan agar fitur yang memiliki skala berbeda dapat dibandingkan secara adil dalam algoritma machine learning, khususnya yang sensitif terhadap skala seperti KNN dan SVM.

3. **Pemilihan Model Machine Learning**:
   Berbagai algoritma machine learning akan dicoba untuk memprediksi kualitas apel, termasuk:
   - **K-Nearest Neighbor (KNN)**: Algoritma ini mengklasifikasikan data berdasarkan tetangga terdekatnya. KNN bergantung pada metrik jarak dan cocok untuk dataset yang memiliki struktur kelas yang jelas.[[4](#)]
   - **Random Forest**: Algoritma ensemble yang memanfaatkan banyak decision tree untuk meningkatkan akurasi prediksi. Random Forest terkenal karena kemampuannya untuk menangani dataset dengan banyak fitur dan mengurangi overfitting.[[5](#)]
   - **Support Vector Machine (SVM)**: Algoritma yang bertujuan untuk menemukan hyperplane optimal untuk memisahkan kelas data dalam ruang multidimensi. SVM dapat digunakan untuk tugas klasifikasi dan regresi, terutama untuk dataset dengan dimensi tinggi.[[6](#)]
   - **Naive Bayes**: Model probabilistik berdasarkan teorema Bayes, yang digunakan untuk klasifikasi dengan asumsi bahwa fitur bersifat independen satu sama lain. Algoritma ini bekerja baik pada dataset yang memiliki distribusi probabilitas yang jelas.[[7](#)]
   - **Extra Trees Classifier**: Sebuah varian dari Random Forest yang menggunakan random splitting pada node-nya untuk meningkatkan generalisasi model. Algoritma ini dikenal memiliki kecepatan dan kinerja yang baik pada data klasifikasi yang kompleks.[[8](#)]

Setiap model akan diuji, dan **evaluasi model** akan dilakukan untuk memilih model dengan **akurasi terbaik** dalam memprediksi kualitas apel.

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

Dataset ini berisi informasi penting yang dibutuhkan untuk membangun model prediksi kualitas apel berdasarkan faktor-faktor visual dan sensorik yang relevan. Data tersebut berasal dari sebuah perusahaan pertanian di Amerika Serikat dan telah diunggah secara publik di platform **Kaggle** dengan judul "_Apple Quality_". Dataset ini membuka peluang untuk melakukan eksplorasi mendalam dalam hal pemahaman kualitas apel, serta membantu pengembangan model machine learning yang dapat mendukung petani dan distributor dalam meningkatkan nilai jual dan kualitas apel mereka.

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

![1  Univariate](https://github.com/user-attachments/assets/14af8aa3-05b3-4ce7-8742-a635db7cd5e2)


- Mengurangi rata-rata (mean) dari setiap data point.
- Membagi hasil pengurangan tersebut dengan standar deviasi data.

Misalnya, variabel **"Size"** memiliki mean -0.51, tetapi standar deviasi tidak diketahui. Namun, berdasarkan nilai minimum (-2) dan maksimum (2), dapat diasumsikan bahwa data telah distandarisasi sehingga memiliki mean 0 dan standar deviasi 1. Hal ini juga berlaku untuk variabel numerik lainnya seperti **"Weight", "Sweetness", "Crunchiness", "Juiciness", "Ripeness"**, dan **"Acidity"**.

### EDA - Multivariate Analysis

![2  Multivariate](https://github.com/user-attachments/assets/223f94ca-6e2b-4da3-a11b-03292c625848)

- **Gambar 2a. Analisis Multivariat**:
  Menggunakan _pairplot_ dari _library Seaborn_, terlihat bahwa pola pasangan antar variabel tampak acak. Salah satu pola yang menarik adalah antara **"Size"** dan **"Sweetness"**, di mana terlihat korelasi negatif: semakin kecil ukuran apel, semakin manis rasanya.

  ![2 1 correlation matrix](https://github.com/user-attachments/assets/8515b554-6069-4223-a104-9ddbf2212426)


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

Pada proses analisis, pengecekan terhadap data missing value menjadi langkah penting. Kehadiran missing values dalam dataset yang kompleks dapat berdampak signifikan terhadap kinerja model, terutama jika data yang hilang merupakan faktor penting dalam menentukan hasil prediksi. Oleh karena itu, sebelum melanjutkan ke tahap pemodelan, sangat krusial untuk mengidentifikasi dan menangani missing values secara tepat, seperti melalui teknik imputation, penghapusan baris atau kolom yang hilang, atau metode lain yang sesuai dengan sifat data. Penanganan yang baik akan membantu meningkatkan kualitas prediksi model.

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
Dalam tahap evaluasi, metrik utama yang digunakan adalah **accuracy**. Akurasi mengukur sejauh mana prediksi yang dilakukan oleh model sesuai dengan nilai aktual dalam data uji. Rumus untuk menghitung akurasi adalah sebagai berikut:

\[
\text{Akurasi} = \frac{\text{Jumlah Prediksi Benar}}{\text{Jumlah Total Data}} \times 100\%
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
