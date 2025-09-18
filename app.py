import os
from dotenv import load_dotenv
import warnings
from flask import Flask, request, jsonify
import joblib
from openai import OpenAI
import spacy

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

def preprocess_text(text: str) -> str:
    """
    Preprocessing teks untuk konsistensi analisis
    """
    if not text:
        return ""
    
    # Normalisasi teks
    text = text.lower().strip()
    
    # Normalisasi kata umum
    replacements = {
        'konslet': 'korsleting',
        'korslet': 'korsleting',
        'listrik': 'listrik',
        'pdam': 'air',
        'keran': 'air',
        'kran': 'air',
        'kebakar': 'kebakaran',
        'kebakaran': 'kebakaran',
        'led': 'korsleting',
        'short': 'korsleting',
        'mati': 'padam',
        'gelap': 'padam',
        'blackout': 'padam'
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    return text

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
                    "content": "Kamu adalah model klasifikasi aduan. Jawab hanya 'Prioritas' atau 'Reguler'. Perhatikan kata-kata seperti 'sudah selesai', 'telah ditangani', 'berhasil diatasi' yang menandakan tidak perlu prioritas."
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
    Aturan manual yang komprehensif untuk override prediksi model/ChatGPT
    berdasarkan kata kunci dan konteks spesifik.
    """
    text_lower = preprocess_text(text)
    
    # ===== KASUS YANG HARUSNYA REGULER =====
    
    # 1. Kebakaran kecil atau sudah ditangani
    if "kebakaran" in text_lower:
        if any(word in text_lower for word in ["kecil", "sampah", "rumput", "ranting", "daun"]):
            return "Reguler"
        if any(word in text_lower for word in ["sudah", "telah", "selesai", "berhasil", "ditangani", "diatasi", "padam", "lampu led"]):
            return "Reguler"
    
    # 2. Masalah listrik non-darurat
    if any(word in text_lower for word in ["listrik", "tegangan", "voltase", "korsleting"]):
        if any(word in text_lower for word in ["perbaikan", "perawatan", "pemeliharaan", "rencana", "jadwal"]):
            return "Reguler"
        if "lampu" in text_lower and ("led" in text_lower or "pendek" in text_lower):
            return "Reguler"
    
    # 3. Masalah air yang tidak darurat
    if any(word in text_lower for word in ["air", "pdam", "keran", "pipa", "bocor"]):
        if any(word in text_lower for word in ["kecil", "sedikit", "tetes", "rembes"]):
            return "Reguler"
        if any(word in text_lower for word in ["perbaikan", "pemeliharaan", "jadwal", "pemutusan"]):
            return "Reguler"
    
    # 4. Laporan yang sudah ditangani atau tidak urgensi
    if any(phrase in text_lower for phrase in [
        "sudah ditangani", "telah selesai", "berhasil diatasi", 
        "tidak perlu", "bukan urgensi", "bukan darurat",
        "hanya informasi", "sekedar laporan", "laporan rutin",
        "sudah beres", "telah diperbaiki", "selesai diperbaiki"
    ]):
        return "Reguler"
    
    # 5. Permintaan informasi atau administratif
    if any(word in text_lower for word in ["informasi", "tanya", "bertanya", "konsultasi", "admin", "administratif", "bertanya"]):
        return "Reguler"
    
    # 6. Keluhan kecil atau maintenance rutin
    if any(word in text_lower for word in ["lubang", "jalan", "trotoar"]):
        if any(word in text_lower for word in ["kecil", "sedang", "kecill", "sedang"]):
            return "Reguler"
    
    # 7. Complain umum non-darurat
    if any(word in text_lower for word in ["complain", "keluhan", "aspirasi", "saran", "masukan"]):
        return "Reguler"
    
    # ===== KASUS YANG HARUSNYA PRIORITAS =====
    
    # 1. Kebakaran aktif dan berbahaya
    if "kebakaran" in text_lower:
        if any(word in text_lower for word in ["besar", "gedung", "rumah", "pabrik", "aktif", "masih", "sedang", "membara"]):
            return "Prioritas"
        if any(word in text_lower for word in ["korban", "luka", "tertrap", "terjebak", "terperangkap"]):
            return "Prioritas"
    
    # 2. Kecelakaan dan darurat medis
    if any(word in text_lower for word in ["kecelakaan", "tabrakan", "benturan", "laka"]):
        return "Prioritas"
    
    if any(word in text_lower for word in ["sakit", "pingsan", "stroke", "serangan", "jantung", "sesak", "napas", "darah"]):
        if any(word in text_lower for word in ["mendadak", "darurat", "gawat", "parah"]):
            return "Prioritas"
    
    # 3. Kejahatan dan kekerasan
    if any(word in text_lower for word in ["maling", "pencuri", "curi", "rampok", "perampok", "ancaman", "teror", "kekerasan", "pembunuh"]):
        return "Prioritas"
    
    # 4. Bencana alam
    if any(word in text_lower for word in ["banjir", "longsor", "gempa", "tsunami", "angin", "puting", "beliung", "topan"]):
        return "Prioritas"
    
    # 5. Kesehatan masyarakat
    if any(word in text_lower for word in ["keracunan", "wabah", "virus", "covid", "demam", "berdarah", "kerumunan"]):
        return "Prioritas"
    
    # 6. Infrastruktur kritis
    if any(word in text_lower for word in ["jembatan", "roboh", "ambruk", "retak", "bahaya", "rubuh", "ambrol"]):
        return "Prioritas"
    
    if "jalan" in text_lower and any(word in text_lower for word in ["rusak", "berat", "besar", "dalam", "berbahaya"]):
        return "Prioritas"
    
    # 7. Keterlibatan orang/nyawa
    if any(word in text_lower for word in ["terjebak", "tertrap", "tersangkut", "terperangkap", "tertimpa"]):
        return "Prioritas"
    
    # 8. Listrik padam luas
    if "listrik" in text_lower and any(word in text_lower for word in ["padam", "mati", "blackout"]):
        if any(word in text_lower for word in ["seluruh", "wilayah", "kota", "desa", "kampung", "area", "komplek"]):
            return "Prioritas"
    
    # 9. Waktu kritis
    if any(word in text_lower for word in ["sekarang", "saat ini", "sedang terjadi", "langsung", "segera", "darurat"]):
        if any(word in text_lower for word in ["kebakaran", "kecelakaan", "bencana", "kriminal"]):
            return "Prioritas"
    
    # 10. Gas berbahaya
    if any(word in text_lower for word in ["gas", "bocor", "meledak", "explosi", "kebocoran"]):
        return "Prioritas"
    
    return None  # Tidak ada override, kembalikan ke prediksi model utama

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

        # Preprocessing teks input
        processed_text = preprocess_text(text)
        
        # 🔹 Cek aturan manual override dengan teks yang sudah diproses
        override = rule_based_override(processed_text)
        if override:
            return jsonify({
                'text': text,
                'prediction': override,
                'prioritas_probability': '100.00%',
                'reguler_probability': '0.00%',
                'model_version': MODEL_VERSION,
                'message': 'Prediksi ditentukan oleh aturan manual (rule-based override).'
            })

        # 🔹 Jika tidak ada override, gunakan ChatGPT
        prediction = get_chatgpt_prediction(text)
        message = "Prediksi utama oleh ChatGPT"

        # 🔹 Jalankan model lokal untuk probabilitas tambahan (info pendukung)
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
            'text': text,
            'prediction': prediction,
            'prioritas_probability': f"{prioritas_proba:.2%}",
            'reguler_probability': f"{reguler_proba:.2%}",
            'model_version': MODEL_VERSION,
            'message': message
        })

    except Exception as e:
        app.logger.error(f"Terjadi kesalahan saat prediksi: {str(e)}")
        return jsonify({'error': 'Terjadi kesalahan saat prediksi.', 'details': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint untuk health check"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'vectorizer_loaded': vectorizer is not None,
        'model_version': MODEL_VERSION
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)