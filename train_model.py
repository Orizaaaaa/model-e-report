import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Unduh stopwords NLTK
nltk.download('stopwords')

# Gabungkan stopwords NLTK dengan custom
base_stopwords = stopwords.words('indonesian')
custom_stopwords = [
    'kepada', 'tersebut', 'sebuah', 'dua', 'ia', 
    'dapat', 'oleh', 'adalah', 'jika', 'lebih'
]
indonesian_stop_words = list(set(base_stopwords + custom_stopwords))

# Load dataset dengan handling encoding
try:
    df = pd.read_csv('data/dataset_pengaduan.csv', encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv('data/dataset_pengaduan.csv', encoding='latin-1')

# Preprocessing dengan optimasi
vectorizer = CountVectorizer(
    ngram_range=(1, 3),  # Mencakup trigram
    stop_words=indonesian_stop_words,
    max_features=1000,  # Diperbesar untuk menangkap lebih banyak fitur
    min_df=2,  # Abaikan kata yang muncul kurang dari 2 dokumen
    lowercase=True,
    decode_error='replace'
)

# Split dataset dengan validasi
X = vectorizer.fit_transform(df['laporan'])
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# Training model dengan optimasi hyperparameter
model = MultinomialNB(
    alpha=0.5,  # Nilai optimal berdasarkan tuning
    fit_prior=False
)
model.fit(X_train, y_train)

# Evaluasi komprehensif
y_pred = model.predict(X_test)
print(f"Akurasi: {accuracy_score(y_test, y_pred):.2%}")
print("\nLaporan Klasifikasi Detil:")
print(classification_report(y_test, y_pred, target_names=['Prioritas', 'Reguler']))

# Simpan model dengan versioning
MODEL_VERSION = "v2.1"
joblib.dump(model, f'model/model_pengaduan_{MODEL_VERSION}.pkl')
joblib.dump(vectorizer, f'model/vectorizer_pengaduan_{MODEL_VERSION}.pkl')

# Contoh prediksi
test_samples = [
    "banyak sampah yang berceceran dekat masjid",
    "ada tiang listrik yang mau roboh di komplek perumahan",
    "truk kecelakaan belum di evakuasi menyebabkan kemacetan"
]

print("\nContoh Prediksi:")
for text in test_samples:
    vec_text = vectorizer.transform([text])
    proba = model.predict_proba(vec_text)[0]
    print(f"\nLaporan: {text}")
    print(f"Prediksi: {model.predict(vec_text)[0]}")
    print(f"Probabilitas: Prioritas ({proba[0]:.2%}) | Reguler ({proba[1]:.2%})")