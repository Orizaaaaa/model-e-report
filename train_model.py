# train_model.py
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
from platformdirs import user_data_dir
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType

# Download NLTK resources
nltk.download('stopwords')

# Konfigurasi
APP_NAME = "model-e-report"
MODEL_VERSION = "v2.1"
model_dir = os.path.join(user_data_dir(APP_NAME), "models")
os.makedirs(model_dir, exist_ok=True)

# Custom stopwords Indonesia
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

# Inisialisasi vectorizer dengan parameter optimal
vectorizer = CountVectorizer(
    ngram_range=(1, 3),
    stop_words=indonesian_stop_words,
    max_features=1000,
    min_df=2,
    lowercase=True,
    decode_error='replace'
)

# Preprocessing data
X = vectorizer.fit_transform(df['laporan'])
y = df['label']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# Training model dengan hyperparameter optimal
model = MultinomialNB(
    alpha=0.5,
    fit_prior=False
)
model.fit(X_train, y_train)

# Evaluasi model
y_pred = model.predict(X_test)
print(f"Akurasi: {accuracy_score(y_test, y_pred):.2%}")
print("\nLaporan Klasifikasi:")
print(classification_report(y_test, y_pred, target_names=['Prioritas', 'Reguler']))

# Simpan model dengan kompresi
joblib.dump(
    model,
    os.path.join(model_dir, f'model_pengaduan_{MODEL_VERSION}.pkl'),
    compress=('gzip', 3),
    protocol=4
)

joblib.dump(
    vectorizer,
    os.path.join(model_dir, f'vectorizer_pengaduan_{MODEL_VERSION}.pkl'),
    compress=('gzip', 3),
    protocol=4
)

# Konversi ke ONNX (opsional)
initial_type = [('text_input', StringTensorType([None, 1]))]
onnx_model = convert_sklearn(
    model,
    initial_types=initial_type,
    doc_string="Model klasifikasi prioritas pengaduan",
    target_opset=12
)

onnx_path = os.path.join(model_dir, f'model_pengaduan_{MODEL_VERSION}.onnx')
with open(onnx_path, "wb") as f:
    f.write(onnx_model.SerializeToString())

print(f"\nModel berhasil disimpan di: {model_dir}")
print(f"Ukuran file model: {os.path.getsize(os.path.join(model_dir, f'model_pengaduan_{MODEL_VERSION}.pkl')) / 1024 / 1024:.2f} MB")
print(f"Ukuran file ONNX: {os.path.getsize(onnx_path) / 1024 / 1024:.2f} MB")

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