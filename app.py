import os
from dotenv import load_dotenv
import warnings
from flask import Flask, request, jsonify
import joblib
from openai import OpenAI  # Import OpenAI client
import spacy  # Untuk deteksi kata kerja aktif

warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)

# Lokasi base directory dan folder model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')

# Versi model sesuai file di folder kamu
MODEL_VERSION = "v3.0"

model_filename = f'model_pengaduan_tfidf_{MODEL_VERSION}.pkl'
vectorizer_filename = f'vectorizer_pengaduan_tfidf_{MODEL_VERSION}.pkl'

model_path = os.path.join(MODEL_DIR, model_filename)
vectorizer_path = os.path.join(MODEL_DIR, vectorizer_filename)

model = None
vectorizer = None

# Load file .env
load_dotenv()
# Inisialisasi client OpenAI
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Load spaCy model untuk deteksi kata kerja
nlp = spacy.load("en_core_web_sm")

def contains_active_verb(text: str) -> bool:
    """
    Fungsi untuk memeriksa apakah teks mengandung kata kerja aktif.
    """
    doc = nlp(text)
    for token in doc:
        if token.pos_ == "VERB":
            return True
    return False

def get_chatgpt_prediction(text: str) -> str:
    """
    Gunakan ChatGPT untuk klasifikasi aduan.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Kamu adalah model klasifikasi aduan. Jawab hanya 'Prioritas' atau 'Reguler'."
                },
                {
                    "role": "user",
                    "content": f"Laporan aduan: {text}"
                }
            ]
        )
        result = response.choices[0].message.content.strip()
        if result not in ["Prioritas", "Reguler"]:
            return "Reguler"  # fallback jika jawaban ChatGPT tidak sesuai
        return result
    except Exception as e:
        print(f"Error dari ChatGPT: {e}")
        return "Reguler"

def rule_based_override(text: str) -> str:
    """
    Aturan manual: jika ada kata 'kebakaran' + 'kecil' atau 'sampah',
    maka dianggap Reguler meskipun ChatGPT / model bilang Prioritas.
    """
    text_lower = text.lower()
    if "kebakaran" in text_lower and ("kecil" in text_lower or "sampah" in text_lower):
        return "Reguler"
    return None

# Coba load model utama
try:
    print(f"Mencoba memuat model dari: {model_path}")
    print(f"Mencoba memuat vectorizer dari: {vectorizer_path}")
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    app.logger.info("Model dan Vectorizer berhasil dimuat.")
except Exception as e:
    print(f"Gagal memuat model atau vectorizer: {str(e)}")
    app.logger.error(f"Gagal memuat model atau vectorizer: {str(e)}")
    try:
        model_filename_fallback = f'model_pengaduan_count_{MODEL_VERSION}.pkl'
        vectorizer_filename_fallback = f'vectorizer_pengaduan_count_{MODEL_VERSION}.pkl'
        model_path_fallback = os.path.join(MODEL_DIR, model_filename_fallback)
        vectorizer_path_fallback = os.path.join(MODEL_DIR, vectorizer_filename_fallback)
        
        print(f"Mencoba memuat model fallback (CountVectorizer) dari: {model_path_fallback}")
        print(f"Mencoba memuat vectorizer fallback (CountVectorizer) dari: {vectorizer_path_fallback}")
        model = joblib.load(model_path_fallback)
        vectorizer = joblib.load(vectorizer_path_fallback)
        app.logger.info("Model CountVectorizer dan Vectorizer berhasil dimuat sebagai fallback.")
    except Exception as fallback_e:
        print(f"Gagal memuat model fallback: {str(fallback_e)}")
        app.logger.error(f"Gagal memuat model fallback: {str(fallback_e)}")
        model = None
        vectorizer = None

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or vectorizer is None:
        return jsonify({'error': 'Inisialisasi model gagal. Silakan periksa log server.'}), 500
    
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'Input teks diperlukan.'}), 400

        # 🔹 Utamakan hasil ChatGPT
        prediction = get_chatgpt_prediction(text)
        message = "Prediksi utama berasal dari ChatGPT."

        # 🔹 Cek aturan manual override
        override = rule_based_override(text)
        if override:
            prediction = override
            message = "Prediksi ditentukan oleh aturan manual (rule-based override)."

        # 🔹 Jalankan model lokal hanya untuk probabilitas tambahan (info pendukung)
        prioritas_proba = 0.0
        reguler_proba = 0.0
        vec_text = vectorizer.transform([text])
        if vec_text.nnz > 0:
            proba = model.predict_proba(vec_text)[0]
            prioritas_idx = list(model.classes_).index("Prioritas") if "Prioritas" in model.classes_ else -1
            reguler_idx = list(model.classes_).index("Reguler") if "Reguler" in model.classes_ else -1

            prioritas_proba = proba[prioritas_idx] if prioritas_idx != -1 else 0.0
            reguler_proba = proba[reguler_idx] if reguler_idx != -1 else 0.0

        return jsonify({
            'prediction': prediction,
            'prioritas_probability': f"{prioritas_proba:.2%}",
            'reguler_probability': f"{reguler_proba:.2%}",
            'model_version': MODEL_VERSION,
            'message': message
        })

    except Exception as e:
        app.logger.error(f"Terjadi kesalahan saat prediksi: {str(e)}")
        return jsonify({'error': 'Terjadi kesalahan saat prediksi.', 'details': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
