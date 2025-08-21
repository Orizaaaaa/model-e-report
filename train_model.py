import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Buat direktori 'model' jika belum ada
os.makedirs('model', exist_ok=True)

# Definisikan daftar stopword Bahasa Indonesia kustom yang komprehensif
indonesian_stop_words = [
    'yang', 'untuk', 'pada', 'ke', 'ini', 'itu', 'adalah', 'akan', 'dan', 'dengan',
    'dari', 'di', 'dalam', 'oleh', 'sebagai', 'tidak', 'atau', 'sudah', 'telah',
    'bisa', 'dapat', 'harus', 'saat', 'para', 'sangat', 'agar', 'namun', 'juga',
    'tersebut', 'kepada', 'sebuah', 'dua', 'ia', 'lebih', 'nya', 'pun', 'lah', 'kah', 
    'saja', 'kok', 'deh', 'dong', 'atas', 'bawah', 'depan', 'belakang', 'kiri', 'kanan', 
    'luar', 'dalam', 'antara', 'melalui', 'tentang', 'mengenai', 'setelah', 'sebelum', 
    'hingga', 'sampai', 'sejak', 'selama', 'meskipun', 'walaupun', 'jika', 'kalau', 'maka',
    'karena', 'sebab', 'meski', 'tetapi', 'tetap', 'bagaimana', 'mengapa', 'siapa', 'dimana', 
    'kemana', 'kapan', 'bagaimanapun', 'apalagi', 'lain', 'lainnya', 'begitu', 'seperti', 
    'yaitu', 'yakni', 'adapun', 'bahkan', 'terutama', 'secara', 'umum', 'biasa', 'setiap', 
    'seluruh', 'tanpa', 'guna', 'bagi', 'macam', 'jenis', 'sekitar', 'kira-kira', 'hanya', 
    'cuma', 'pasti', 'benar', 'paling', 'kurang', 'banyak', 'sedikit', 'cukup', 'belum', 
    'masih', 'segera', 'nantinya', 'kemudian', 'akhirnya', 'awalnya', 'pertama', 'kedua', 
    'ketiga', 'selain', 'demikian', 'begitulah', 'berikutnya', 'selanjutnya', 'olehnya', 
    'padanya', 'kepadanya', 'daripada', 'daripadanya', 'denganmu', 'untukmu', 'bagiku', 'bagimu', 
    'baginya', 'milikku', 'milikmu', 'miliknya', 'aku', 'saya', 'kami', 'kita', 'anda', 'kamu', 
    'mereka', 'dia', 'beliau', 'diriku', 'dirimu', 'dirinya', 'sendiri', 'diri', 'semua', 'segala', 
    'tiap', 'beberapa', 'seluruh', 'keseluruhan', 'sebagian', 'sering', 'jarang', 'kadang', 'senantiasa', 
    'seraya', 'ketika', 'sementara', 'serta', 'bersama', 'kecuali', 'bahkan', 'lagi', 'jua', 'toh', 
    'wah', 'duh', 'ah', 'eh', 'oh', 'ya', 'ayo', 'mari', 'silakan', 'tolong', 'maaf', 'terima', 
    'kasih', 'sama', 'sesuai', 'cocok', 'pantas', 'layak', 'mungkin', 'barangkali', 'seandainya', 
    'andaikata', 'sekiranya', 'apabila', 'bilamana', 'manakala', 'kendati', 'biarpun', 'sekalipun', 
    'memang', 'betul', 'nyata', 'sungguh', 'amat', 'terlalu', 'begitu', 'demikian', 'kian', 'makin', 
    'pula', 'serta', 'maupun', 'apalagi', 'disamping', 'ditambah', 'sambil', 'sewaktu', 'sesudah', 
    'sejak', 'begitupun', 'begini', 'bak', 'laksana', 'bagai', 'buat', 'guna', 'sebab', 'karena', 
    'akibat', 'lantaran', 'sehingga', 'maka', 'demi', 'beserta', 'diantara', 'diluar', 'diatas', 
    'dibawah', 'dihadapan', 'dibelakang', 'disini', 'disana', 'dimana', 'kemana', 'darimana', 'entah', 
    'entahlah', 'jikalau', 'kendatipun', 'sungguhpun', 'adalah', 'ialah', 'merupakan', 'ibarat', 
    'umpama', 'bukan', 'jangan', 'bukanlah', 'tiada', 'kan', 'tuh', 'yuk'
]

# Load and preprocess the dataset
df = pd.read_csv('data/dataset.csv', encoding='utf-8')

# Membersihkan data: menghapus duplikasi dan data kosong
df_cleaned = df.dropna().drop_duplicates()

# Gunakan TfidfVectorizer untuk representasi teks
tfidf_vectorizer = TfidfVectorizer(
    ngram_range=(1, 3),  # Mencakup unigram, bigram, trigram
    stop_words=indonesian_stop_words,
    max_features=1000,   # Jumlah fitur maksimal
    min_df=2,            # Abaikan kata yang muncul di kurang dari 2 dokumen
    lowercase=True,
    decode_error='replace'
)

#  Preprocessing Data
X_tfidf = tfidf_vectorizer.fit_transform(df_cleaned['laporan'])
y = df_cleaned['label']

# Split data menjadi pelatihan dan pengujian
X_train_tfidf, X_test_tfidf, y_train_tfidf, y_test_tfidf = train_test_split(
    X_tfidf, y,
    test_size=0.2,  # 20% untuk pengujian
    stratify=y,     # Pastikan distribusi kelas sama
    random_state=42 # Untuk reprodusibilitas
)

# Latih model Naive Bayes
model_tfidf = MultinomialNB(alpha=0.1, fit_prior=False)
model_tfidf.fit(X_train_tfidf, y_train_tfidf)

# Evaluasi model
y_pred_tfidf = model_tfidf.predict(X_test_tfidf)
accuracy = accuracy_score(y_test_tfidf, y_pred_tfidf)
classification_rep = classification_report(y_test_tfidf, y_pred_tfidf, target_names=['Prioritas', 'Reguler'])

# Simpan model dan vectorizer
MODEL_VERSION = "v3.0"
joblib.dump(model_tfidf, f'model/model_pengaduan_tfidf_{MODEL_VERSION}.pkl')
joblib.dump(tfidf_vectorizer, f'model/vectorizer_pengaduan_tfidf_{MODEL_VERSION}.pkl')


# === Confusion Matrix ===
cm = confusion_matrix(y_test_tfidf, y_pred_tfidf, labels=model_tfidf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model_tfidf.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Naive Bayes")
plt.show()

# === Grafik Akurasi Sederhana ===
plt.bar(["Akurasi Model"], [accuracy])
plt.ylim(0,1)
plt.ylabel("Akurasi")
plt.title("Akurasi Model Naive Bayes")
plt.text(0, accuracy + 0.02, f"{accuracy:.2%}", ha='center')
plt.show()


print("Jumlah data asli:", len(df))
print("Jumlah data setelah dibersihkan:", len(df_cleaned))
print("Jumlah data latih:", len(X_train_tfidf.toarray()))
print("Jumlah data uji:", len(X_test_tfidf.toarray()))


# Print hasil evaluasi
print(f"\nAkurasi 🎯 : {accuracy:.2%}")
print("\nLaporan Klasifikasi Detail:")
print(classification_rep)
