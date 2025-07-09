import pandas as pd
import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
import numpy as np

# Buat direktori 'model' jika belum ada
os.makedirs('model', exist_ok=True)

# Definisikan daftar stopword Bahasa Indonesia kustom yang komprehensif
# Ini menggantikan penggunaan nltk.corpus.stopwords untuk menghindari masalah unduhan
indonesian_stop_words = [
    'yang', 'untuk', 'pada', 'ke', 'ini', 'itu', 'adalah', 'akan', 'dan', 'dengan',
    'dari', 'di', 'dalam', 'oleh', 'sebagai', 'tidak', 'atau', 'sudah', 'telah',
    'bisa', 'dapat', 'harus', 'saat', 'para', 'sangat', 'agar', 'namun', 'juga',
    'tersebut', 'kepada', 'sebuah', 'dua', 'ia', 'lebih', # Stopwords dari input user
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
    'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', # Huruf tunggal
    'nya', 'pun', 'lah', 'kah', 'saja', 'kok', 'deh', 'dong', # Partikel umum
    'atas', 'bawah', 'depan', 'belakang', 'kiri', 'kanan', 'luar', 'dalam',
    'antara', 'melalui', 'tentang', 'mengenai', 'setelah', 'sebelum', 'hingga',
    'sampai', 'sejak', 'selama', 'meskipun', 'walaupun', 'jika', 'kalau', 'maka',
    'karena', 'sebab', 'meski', 'tetapi', 'tetap', 'bagaimana', 'mengapa',
    'siapa', 'dimana', 'kemana', 'kapan', 'bagaimanapun', 'apalagi', 'lain',
    'lainnya', 'begitu', 'seperti', 'yaitu', 'yakni', 'adapun', 'bahkan',
    'terutama', 'secara', 'umum', 'biasa', 'setiap', 'seluruh', 'tanpa',
    'guna', 'bagi', 'macam', 'jenis', 'sekitar', 'kira-kira', 'hanya', 'cuma',
    'pasti', 'benar', 'paling', 'kurang', 'banyak', 'sedikit',
    'cukup', 'belum', 'masih', 'segera', 'nantinya',
    'kemudian', 'akhirnya', 'awalnya', 'pertama', 'kedua', 'ketiga',
    'selain', 'demikian', 'begitulah', 'berikutnya', 'selanjutnya',
    'olehnya', 'padanya', 'kepadanya', 'daripada', 'daripadanya', 'denganmu',
    'untukmu', 'bagiku', 'bagimu', 'baginya', 'milikku', 'milikmu',
    'miliknya', 'aku', 'saya', 'kami', 'kita', 'anda', 'kamu', 'mereka', 'dia',
    'beliau', 'diriku', 'dirimu', 'dirinya', 'sendiri', 'diri', 'semua', 'segala',
    'tiap', 'beberapa', 'seluruh', 'keseluruhan', 'sebagian', 'sering', 'jarang',
    'kadang', 'senantiasa', 'seraya', 'ketika', 'sementara', 'serta', 'bersama',
    'kecuali', 'bahkan', 'lagi', 'jua', 'toh', 'wah', 'duh', 'ah',
    'eh', 'oh', 'ya', 'ayo', 'mari', 'silakan', 'tolong', 'maaf', 'terima',
    'kasih', 'sama', 'sesuai', 'cocok', 'pantas', 'layak', 'mungkin', 'barangkali',
    'seandainya', 'andaikata', 'sekiranya', 'apabila', 'bilamana', 'manakala',
    'kendati', 'biarpun', 'sekalipun', 'memang', 'betul',
    'nyata', 'sungguh', 'amat', 'terlalu', 'begitu',
    'demikian', 'kian', 'makin', 'pula', 'serta', 'maupun',
    'apalagi', 'disamping', 'ditambah', 'sambil', 'sewaktu',
    'sesudah', 'sejak', 'begitupun', 'begini',
    'bak', 'laksana', 'bagai', 'buat', 'guna', 'sebab', 'karena',
    'akibat', 'lantaran', 'sehingga', 'maka', 'demi',
    'beserta', 'diantara', 'diluar', 'diatas', 'dibawah', 'dihadapan', 'dibelakang',
    'disini', 'disana', 'dimana', 'kemana', 'darimana', 'entah', 'entahlah',
    'jikalau', 'kendatipun', 'sungguhpun',
    'adalah', 'ialah', 'merupakan', 'ibarat', 'umpama',
    'bukan', 'jangan', 'bukanlah', 'tiada', 'kan', 'tuh', 'yuk'
]

# Load dataset dengan handling encoding
try:
    # Menggunakan nama file yang diunggah: dataset_pengaduan_1000.csv
    df = pd.read_csv('data/dataset.csv', encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv('data/dataset.csv', encoding='latin-1')

# Pastikan kolom 'laporan' dan 'label' ada di dataset
if 'laporan' not in df.columns or 'label' not in df.columns:
    print("Error: Dataset harus memiliki kolom 'laporan' dan 'label'.")
    exit()

# --- Menggunakan CountVectorizer (sesuai permintaan awal user) ---
print("--- Melatih Model Menggunakan CountVectorizer ---")
count_vectorizer = CountVectorizer(
    ngram_range=(1, 3),  # Mencakup unigram, bigram, dan trigram
    stop_words=indonesian_stop_words, # Menggunakan stopword kustom
    max_features=1000,   # Jumlah fitur maksimum yang akan dipertimbangkan
    min_df=2,            # Abaikan kata yang muncul kurang dari 2 dokumen
    lowercase=True,      # Ubah semua teks menjadi huruf kecil
    decode_error='replace' # Tangani error decoding karakter
)

# Transformasi teks laporan menjadi fitur numerik
X_count = count_vectorizer.fit_transform(df['laporan'])
y = df['label']

# Bagi dataset menjadi data training dan testing
X_train_count, X_test_count, y_train_count, y_test_count = train_test_split(
    X_count, y,
    test_size=0.2, # 20% data untuk testing
    stratify=y,    # Pastikan proporsi kelas sama di training dan testing
    random_state=42 # Untuk reproduktibilitas hasil
)

# Inisialisasi dan latih model Multinomial Naive Bayes
model_count = MultinomialNB(
    alpha=0.5,  # Parameter smoothing Laplace/Lidstone. Nilai lebih rendah bisa meningkatkan akurasi.
    fit_prior=False # Jangan mempelajari probabilitas prior kelas.
)
model_count.fit(X_train_count, y_train_count)

# Evaluasi model CountVectorizer
y_pred_count = model_count.predict(X_test_count)
print(f"Akurasi (CountVectorizer): {accuracy_score(y_test_count, y_pred_count):.2%}")
print("\nLaporan Klasifikasi Detil (CountVectorizer):")
print(classification_report(y_test_count, y_pred_count, target_names=['Prioritas', 'Reguler']))

# Simpan model CountVectorizer dan vectorizer
MODEL_VERSION = "v2.3" # Versi model terbaru
joblib.dump(model_count, f'model/model_pengaduan_count_{MODEL_VERSION}.pkl')
joblib.dump(count_vectorizer, f'model/vectorizer_pengaduan_count_{MODEL_VERSION}.pkl')
print(f"Model CountVectorizer dan Vectorizer disimpan sebagai 'model/model_pengaduan_count_{MODEL_VERSION}.pkl' dan 'model/vectorizer_pengaduan_count_{MODEL_VERSION}.pkl'")


# --- Penambahan untuk Akurasi: Implementasi TF-IDF Vectorizer ---
# TF-IDF seringkali memberikan bobot yang lebih baik pada kata-kata penting
print("\n--- Melatih Model Menggunakan TfidfVectorizer (untuk perbandingan akurasi) ---")
tfidf_vectorizer = TfidfVectorizer(
    ngram_range=(1, 3),
    stop_words=indonesian_stop_words,
    max_features=1000,
    min_df=2,
    lowercase=True,
    decode_error='replace'
)

X_tfidf = tfidf_vectorizer.fit_transform(df['laporan'])
X_train_tfidf, X_test_tfidf, y_train_tfidf, y_test_tfidf = train_test_split(
    X_tfidf, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

model_tfidf = MultinomialNB(
    alpha=0.1, # MultinomialNB seringkali berkinerja lebih baik dengan nilai alpha yang lebih kecil untuk TF-IDF
    fit_prior=False
)
model_tfidf.fit(X_train_tfidf, y_train_tfidf)

# Evaluasi model TF-IDF
y_pred_tfidf = model_tfidf.predict(X_test_tfidf)
print(f"Akurasi (TfidfVectorizer): {accuracy_score(y_test_tfidf, y_pred_tfidf):.2%}")
print("\nLaporan Klasifikasi Detil (TfidfVectorizer):")
print(classification_report(y_test_tfidf, y_pred_tfidf, target_names=['Prioritas', 'Reguler']))

# Simpan model TF-IDF dan vectorizer
joblib.dump(model_tfidf, f'model/model_pengaduan_tfidf_{MODEL_VERSION}.pkl')
joblib.dump(tfidf_vectorizer, f'model/vectorizer_pengaduan_tfidf_{MODEL_VERSION}.pkl')
print(f"Model TfidfVectorizer dan Vectorizer disimpan sebagai 'model/model_pengaduan_tfidf_{MODEL_VERSION}.pkl' dan 'model/vectorizer_pengaduan_tfidf_{MODEL_VERSION}.pkl'")


# --- Fungsi Prediksi dengan Penanganan 'Reguler' jika tidak ada di dataset dan menampilkan score ---
def predict_priority(text_sample, model, vectorizer):
    """
    Melakukan prediksi prioritas aduan dan menampilkan probabilitasnya.
    Jika tidak ada kata kunci yang dikenali, default ke 'Reguler'.
    """
    vec_text = vectorizer.transform([text_sample])

    # Periksa apakah vektor yang ditransformasi kosong (semua nol)
    # Ini menunjukkan bahwa tidak ada kata dalam sampel yang ada di vocabulary model
    if vec_text.nnz == 0: # nnz (number of non-zero entries) untuk sparse matrix
        predicted_class = 'Reguler' # Default ke 'Reguler' jika tidak ada fitur yang ditemukan
        
        # Tetapkan probabilitas default untuk kasus ini
        # Mengasumsikan 'Prioritas' dan 'Reguler' adalah kelas yang ada
        prioritas_proba_display = 0.1 # Probabilitas rendah untuk Prioritas
        reguler_proba_display = 0.9  # Probabilitas tinggi untuk Reguler

        print(f"\nLaporan: {text_sample}")
        print(f"Prediksi: {predicted_class} (Tidak ada kata kunci yang dikenali dalam dataset training)")
        print(f"Probabilitas: Prioritas ({prioritas_proba_display:.2%}) | Reguler ({reguler_proba_display:.2%})")
        return predicted_class, [prioritas_proba_display, reguler_proba_display]
    else:
        # Dapatkan prediksi kelas dan probabilitasnya
        predicted_class = model.predict(vec_text)[0]
        proba = model.predict_proba(vec_text)[0]
        
        # Pastikan probabilitas 'Prioritas' dan 'Reguler' ditampilkan dengan benar
        # berdasarkan urutan kelas dalam model.classes_
        prioritas_idx = -1
        reguler_idx = -1
        for i, class_name in enumerate(model.classes_):
            if class_name == 'Prioritas':
                prioritas_idx = i
            elif class_name == 'Reguler':
                reguler_idx = i
        
        prioritas_proba_display = proba[prioritas_idx] if prioritas_idx != -1 else 0
        reguler_proba_display = proba[reguler_idx] if reguler_idx != -1 else 0

        print(f"\nLaporan: {text_sample}")
        print(f"Prediksi: {predicted_class}")
        print(f"Probabilitas: Prioritas ({prioritas_proba_display:.2%}) | Reguler ({reguler_proba_display:.2%})")
        return predicted_class, proba

# Contoh prediksi menggunakan model CountVectorizer
print("\n--- Contoh Prediksi menggunakan CountVectorizer ---")
test_samples = [
    "banyak sampah yang berceceran dekat masjid",
    "ada tiang listrik yang mau roboh di komplek perumahan",
    "truk kecelakaan belum di evakuasi menyebabkan kemacetan",
    "ada kucing nyasar di pohon", # Contoh aduan yang mungkin tidak ada di dataset
    "saya kehilangan dompet", # Contoh aduan yang mungkin tidak ada di dataset
    "ada pohon tumbang menghalangi jalan" # Contoh aduan yang relevan
]

for text in test_samples:
    predict_priority(text, model_count, count_vectorizer)

# Contoh prediksi menggunakan model TfidfVectorizer (untuk perbandingan)
print("\n--- Contoh Prediksi menggunakan TfidfVectorizer ---")

for text in test_samples:
    predict_priority(text, model_tfidf, tfidf_vectorizer)