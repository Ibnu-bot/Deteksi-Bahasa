#Mengimport Pustaka
import pandas as pd
import numpy as np

#Membaca data dari file csv / datashet
data = pd.read_csv("./dataset_bahasa.csv")
print(data.head())

#Mengecek nilai null
data.isnull().sum()

#Menghitung jumlah nilai unik dalam kolom language
data["language"].value_counts()

#Membuat model klasifikasi
x = np.array(data["Text"])
y= np.array(data["language"])

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
cv = CountVectorizer()
X = cv.fit_transform(x)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=41)

model = MultinomialNB()
model.fit(X_train, y_train)

#Mengevaluasi performa model
model.score(X_test, y_test)

#Menerima inputan teks dari user
user = input("Masukan Teks : ")

#Melakukan prediksi menggunakan model
data = cv.transform([user]).toarray()
output = model.predict(data)
print(output)