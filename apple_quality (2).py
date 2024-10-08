# -*- coding: utf-8 -*-
"""Apple_Quality.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1jBNxpfjLLP9EHf2LEkQpfFLKEH6p4zK9

# **Prediksi Kualitas Apel - Izza Auliyai Rabby**

## **Deskripsi Proyek**

### **Deskripsi Latar Belakang Proyek Prediksi Kualitas Apel dengan Machine Learning**


Proyek ini bertujuan untuk mengembangkan model machine learning yang mampu memprediksi kualitas apel secara lebih cepat dan akurat. Proses penentuan kualitas apel yang dilakukan secara manual saat ini membutuhkan banyak waktu dan tenaga, serta rentan terhadap kesalahan manusia. Hal ini menyebabkan kerugian bagi petani dan distributor, serta mempengaruhi kepuasan konsumen. Dengan menggunakan model prediktif, masalah ini dapat diatasi melalui solusi yang lebih efisien, akurat, dan transparan, memberikan hasil yang lebih baik bagi semua pihak terkait.

Model AI dalam konteks ini berfungsi sebagai asisten yang membantu mengotomatisasi dan meningkatkan efisiensi prediksi kualitas apel. Dengan menggunakan machine learning, AI dapat menganalisis data secara lebih cepat dan akurat dibandingkan penilaian manual, mengurangi risiko kesalahan manusia, dan mempercepat proses penentuan kualitas. Hal ini bermanfaat bagi petani, distributor, serta konsumen, karena memberikan hasil yang lebih konsisten dan transparan dalam menentukan kualitas produk pertanian.

## 1. Import Library
"""

!pip install -q kaggle

#Import Load data Library
import os
import numpy as np
from google.colab import files
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile

# Import train test split
from sklearn.model_selection import train_test_split
# Import Minmaxscaler
from sklearn.preprocessing import MinMaxScaler
#Import Model
# KNN
from sklearn.neighbors import KNeighborsClassifier
# Random Forest Classifiers
from sklearn.ensemble import RandomForestClassifier
# Evaluation Matrix
from sklearn.metrics import accuracy_score
# Support Vector Machines
from sklearn.svm import SVC
# Naive Bayes
from sklearn.naive_bayes import BernoulliNB
# Extra Trees Classifiers
from sklearn.ensemble import ExtraTreesClassifier

"""## 2. Data Understanding  

Data Understanding adalah proses untuk memahami informasi yang terkandung dalam data dan mengevaluasi kualitas data tersebut. Hal ini mencakup analisis terhadap struktur, konten, dan relevansi data, serta identifikasi potensi masalah seperti missing values atau outliers yang dapat memengaruhi hasil analisis lebih lanjut.

### 2.1 Data Loading

Data Loading adalah tahap di mana data diimpor atau dimuat ke dalam lingkungan analisis dari berbagai sumber. Proses ini melibatkan pengumpulan data dari file, database, atau API eksternal untuk digunakan dalam pemrosesan lebih lanjut. Pada tahap ini, penting memastikan bahwa data berhasil dimuat dengan benar dan siap untuk dianalisis.
"""

!kaggle datasets download -d nelgiriyewithana/apple-quality

"""## 2.1.2 Detail Datasets

Dataset ini berisi informasi tentang berbagai atribut dari sejumlah buah, memberikan wawasan mengenai karakteristiknya. Data mencakup detail seperti ID buah, ukuran, berat, tingkat kemanisan, kerenyahan, kadar jus, tingkat kematangan, keasaman, dan kualitas keseluruhan.

Dataset ini cocok untuk analisis :

Klasifikasi Buah : Kembangkan model klasifikasi untuk mengelompokkan buah-buahan berdasarkan fitur-fitur yang dimilikinya, seperti ukuran, berat, tingkat kematangan, dan keasaman.

Prediksi Kualitas : Buat model prediksi untuk menilai kualitas buah menggunakan berbagai atribut, seperti kerenyahan, tingkat kemanisan, dan kadar jus, dengan tujuan menghasilkan rating kualitas yang akurat berdasarkan karakteristik tersebut.


"""

zip_ref = zipfile.ZipFile('/content/apple-quality.zip', 'r')
zip_ref.extractall('/content/')
zip_ref.close()

df = pd.read_csv('/content/apple_quality.csv')

"""# **2.2 Exploratory Data Analysis (EDA)**

Exploratory Data Analysis (EDA) dalam Bahasa Indonesia adalah proses awal dalam analisis data untuk memahami struktur, pola, dan hubungan antar variabel di dalam dataset. Melalui EDA, kita dapat mengidentifikasi tren, mendeteksi anomali, serta memeriksa asumsi dasar sebelum melakukan pemodelan.

## **2.2.1 EDA - Deskripsi Variabel**
"""

df.head(10)

"""Dari dataframe di atas kita dapat melihat bahwa pada dataset ini terdapat 9 kolom. Diantaranya:

- `A_id` : Pengidentifikasi unik untuk setiap buah
- `Size` : Ukuran buah
- `Weight` : Berat buah
- `Sweetness` : Tingkat kemanisan buah
- `Crunchiness` : Tekstur yang menunjukkan kerenyahan buah
-` Juiciness` : Tingkat kesegaran buah
- `Ripeness` : Tahap kematangan buah
- `Acidity` : Tingkat keasaman buah
- `Quality` : Kualitas buah secara keseluruhan
"""

df.drop("A_id",axis=1,inplace=True)

"""Dikarenakan kolom `A_id` tidak mempengaruhi model maka akan di drop / dihapus.




"""

df.info()

"""Dari eksekusi method `df.info()` terdapat:

- Terdapat 6 kolom numerik dengan tipe data float64 yaitu: Size, Weight, Sweetness, Crunchiness, Juiciness dan Ripeness.
- Terdapat 2 kolom dengan tipe data object yaitu: Acidity dan Quality.

Namun pada data aslinya kolom ` Acidity` adalah bertipe float64, yang nantinya akan kita rubah.
"""

df.describe()

"""Fungsi `describe()` memberikan informasi statistik pada masing-masing kolom, antara lain:

- `Count` adalah jumlah sampel pada data.
- `Mean` adalah nilai rata-rata.
- `Std` adalah standar deviasi.
- `Min` yaitu nilai minimum setiap kolom.
- `25%` adalah kuartil pertama. Kuartil adalah nilai yang menandai batas interval dalam empat bagian sebaran yang sama.
- `50%` adalah kuartil kedua, atau biasa juga disebut median (nilai tengah).
-` 75%` adalah kuartil ketiga.
- `Max` adalah nilai maksimum.
"""

df.shape

"""Dari eksekusi method` df.shape` Terlihat:
<br>

| Jumlah Baris | Jumlah Kolom |
| ------ | ------ |
| 4001 | 8 |


<br>

## **2.2.2 EDA - Menangani Missing Value dan Outliers**
"""

df.duplicated().sum()

"""Melihat apakah terdapat data yang terduplikat."""

df.Quality.value_counts(normalize=True)

df.isnull().sum()

data_miss = df[df.isnull().any(axis=1)]
data_miss



"""Dapat dilihat terdapat missing value yang mana akan kita hapus."""

df.dropna(inplace=True)
df.isnull().sum().sum()

df.describe()

df["Acidity"] = df["Acidity"].astype("float64")

"""Merubah tipe data kolom `Acidity` menjadi data float64."""

df.info()

"""Dapat kita lihat:
- Jumlah data` Float64` ada 7 dan `object `ada 1.
"""

df.shape

"""Jumlah datasets menjadi `4000` dikarenakan kita telah menghapus missing value.

**Visualisasi Outlier**
"""

df_outlier = df.select_dtypes(exclude=['object'])
for column in df_outlier:
    plt.figure()
    sns.boxplot(data=df_outlier, x=column, color='skyblue')  # Change color here
    plt.title(f'Boxplot of {column}')  # Optional: Add title for clarity
    plt.show()  # Show the plot

"""*Menghapus outliers yang ada pada dataset*  


Pada kasus ini, kita akan mendeteksi outliers dengan teknik visualisasi data (boxplot). Kemudian, menangani outliers dengan teknik IQR method.


```
IQR = Inter Quartile Range
IQR = Q3 - Q1
```


"""

# prompt: IQR = Inter Quartile Range
# IQR = Q3 - Q1

# Calculate the IQR for each numerical feature
for column in df_outlier:
  Q1 = df[column].quantile(0.25)
  Q3 = df[column].quantile(0.75)
  IQR = Q3 - Q1

  # Define the lower and upper bounds for outliers
  lower_bound = Q1 - 1.5 * IQR
  upper_bound = Q3 + 1.5 * IQR

  # Remove outliers
  df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Print the updated DataFrame
print(df)

df.shape

"""Jumlah Datasets setalah kita hapus Outlier: `3758, 8`

## **2.2.3 EDA - Analisis Univariate **

Analisis univariat berfokus pada distribusi dan karakteristik dari setiap variabel dalam dataset. Ini mencakup pemeriksaan frekuensi, tendensi sentral, dan penyebaran dari setiap fitur secara individu. Metode umum yang digunakan termasuk histogram untuk data kontinu dan diagram batang untuk data kategorikal. Analisis ini membantu memahami bagaimana setiap variabel berperilaku sendiri, sebelum mengeksplorasi hubungan antar variabel. Selain itu, analisis ini juga berguna untuk mengidentifikasi outlier atau anomali yang memerlukan penanganan lebih lanjut.
"""

# Membuat histogram dengan warna biru langit
df.hist(bins=50, figsize=(20, 15), color='skyblue')
plt.show()

"""## **2.2.4 EDA - Analisis Multivariate**"""

# Menggunakan pairplot dengan warna biru langit
sns.pairplot(df, diag_kind='kde', plot_kws={'color': 'skyblue'}, diag_kws={'color': 'skyblue'})

# Numerik Kolom Saja
numeric_df = df.select_dtypes(include=[float, int])

# Menghitung Matrik Korelasi
correlation_matrix = numeric_df.corr().round(2)

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(data=correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Matriks Korelasi untuk Fitur Numerik", size=20)
plt.show()

"""# **3. Data Preparation**

Data Preparation adalah langkah penting dalam proses pengembangan model Machine Learning, yang bertujuan untuk menyiapkan data agar siap digunakan dalam pelatihan model. Tahap ini meliputi pembersihan data, pengolahan nilai yang hilang, pengkodean variabel kategorikal, dan normalisasi atau standarisasi fitur. Dengan melakukan persiapan yang tepat, kualitas model yang dihasilkan dapat meningkat, sehingga meningkatkan akurasi dan keandalannya dalam melakukan prediksi.

## **3.1 Data Clening**
"""

df.Quality = (df.Quality == "good").astype(int)  # good:1 , bad:0

x = df.drop("Quality",axis=1)
y = df.Quality

x.shape,y.shape

"""## **3.2 Train-Test-Split**"""

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=60)

print(f'Total datasets: {len(x)}')
print(f'Total data Latih: {len(x_train)}')
print(f'Total data Uji: {len(x_test)}')

"""## **3.3 Normalisasi**"""

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

"""# **4. Model Development**

## **Lazy Predict Library**

**LazyPredict** adalah pustaka Python yang mempermudah dalam memilih model machine learning dengan cepat. Pustaka ini secara otomatis mengevaluasi dan membandingkan berbagai algoritma pembelajaran mesin pada dataset tertentu.

Keuntungan menggunakan LazyPredict:

Efisien dan cepat: LazyPredict dapat mengevaluasi dan membandingkan banyak model secara cepat, menghemat waktu dan tenaga.

Membantu menemukan model terbaik: LazyPredict mempermudah identifikasi model yang memiliki performa terbaik tanpa harus menguji satu per satu secara manual.

Sangat cocok untuk analisis awal: Ideal untuk memulai proyek machine learning tanpa terjebak dalam pemilihan model.

"Simplicity is the ultimate sophistication." - Leonardo da Vinci
"""

!pip install lazypredict

# lazyclassifiyers
!pip install scikit-learn==1.0.2

from lazypredict.Supervised import LazyClassifier
clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
models,predictions = clf.fit(x_train, x_test, y_train, y_test)
print(models)

"""**Visualisasi Model LazyPredict**"""

temp = models.sort_values(by="Accuracy", ascending=True)
plt.figure(figsize=(10, 8))
plt.barh(temp.index, temp["Accuracy"], color='deepskyblue')  # Change color here
plt.xlabel("Akurasi")
plt.title("Akurasi Model Machine Learning")
plt.show()

# Memilih 7 Model saja
models = pd.DataFrame(index=['accuracy_score'],
                      columns=['KNN', 'RandomForest', 'SVM', 'Naive Bayes','Extra trees classifier', 'LogisticRegression', 'XGBClassifiers'])

"""## **4.1 KNN (K-Nearest Neighbor)**"""

model_knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
model_knn.fit(x_train, y_train)

knn_pred = model_knn.predict(x_test)
models.loc['accuracy_score','KNN'] = accuracy_score(y_test, knn_pred)

"""## **4.2 Random Forest**"""

model_rf = RandomForestClassifier(max_depth= 20)
model_rf.fit(x_train, y_train)

rf_pred = model_rf.predict(x_test)
models.loc['accuracy_score','RandomForest'] = accuracy_score(y_test, rf_pred)

"""## **4.3 Support Vector Classifier**"""

model_svc = SVC()
model_svc.fit(x_train, y_train)

svc_pred = model_svc.predict(x_test)
models.loc['accuracy_score','SVM'] = accuracy_score(y_test, svc_pred)

"""### **4.4 Naive Bayes**"""

model_nb = BernoulliNB()
model_nb.fit(x_train, y_train)

nb_pred = model_nb.predict(x_test)
models.loc['accuracy_score','Naive Bayes'] = accuracy_score(y_test, nb_pred)

"""### **4.5 Extra trees classifier**"""

model_etc = ExtraTreesClassifier(n_estimators=100, max_depth= 10,n_jobs= 2,random_state= 100)
model_etc.fit(x_train, y_train)

etc_pred = model_etc.predict(x_test)
models.loc['accuracy_score','Extra trees classifier'] = accuracy_score(y_test, etc_pred)

"""# **5. Evaluasi Model**

## **5.1 Score Model**
"""

# print(models) with description of accuracy and ecaluation metrics, create plots

import matplotlib.pyplot as plt
print(models)
print("Deskripsi Akurasi dan Metrik Evaluasi:")
print("Model KNN memiliki akurasi sebesar:", models.loc['accuracy_score','KNN'])
print("Model Random Forest memiliki akurasi sebesar:", models.loc['accuracy_score','RandomForest'])
print("Model SVM memiliki akurasi sebesar:", models.loc['accuracy_score','SVM'])
print("Model Naive Bayes memiliki akurasi sebesar:", models.loc['accuracy_score','Naive Bayes'])
print("Model Extra Trees Classifier memiliki akurasi sebesar:", models.loc['accuracy_score','Extra trees classifier'])

"""## **5.2 plot Model**"""

# buat plot dan deskripsinya secara holistik

import matplotlib.pyplot as plt
# Create a bar plot for model accuracy
plt.figure(figsize=(10, 6))
plt.bar(models.columns, models.loc['accuracy_score'], color='skyblue')
plt.xlabel("Model")
plt.ylabel("Akurasi")
plt.title("Perbandingan Akurasi Model")
plt.show()


# Description of the plot
print("Deskripsi Plot:")
print("Plot ini menampilkan perbandingan akurasi dari berbagai model machine learning yang telah dievaluasi pada dataset apel. ")
print("Sumbu X menunjukkan model yang digunakan, dan sumbu Y menunjukkan skor akurasi yang dicapai oleh masing-masing model.")
print("Berdasarkan plot ini, kita dapat dengan mudah membandingkan kinerja setiap model dan mengidentifikasi model mana yang memiliki akurasi terbaik dalam memprediksi kualitas apel.")
print("Model dengan akurasi tertinggi dapat digunakan sebagai model terbaik untuk memprediksi kualitas apel pada dataset yang sama atau pada dataset baru yang serupa.")

# Optional: Add more specific descriptions for each model, e.g.
# print("Model KNN menunjukkan akurasi yang cukup baik...")
# print("Model Random Forest menunjukkan akurasi yang sangat baik...")

# Further Analysis and Interpretations
print("\nAnalisis dan Interpretasi:")
print("Berdasarkan perbandingan model, model Random Forest dan Extra trees classifier menunjukkan akurasi yang tinggi dalam memprediksi kualitas apel.")
print("Ini mungkin karena kemampuan model ini untuk menangani kompleksitas data dan membuat keputusan berdasarkan banyak fitur.")
print("Model lainnya seperti KNN, SVM, dan Naive Bayes juga menunjukkan akurasi yang cukup baik, tetapi mungkin kurang optimal dibandingkan Random Forest dan Extra trees classifier dalam dataset ini.")

"""### Mendapatkan Requirement txt"""

!pip freeze > requirements.txt

from google.colab import files
files.download('requirements.txt')