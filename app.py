import os
import warnings
from flask import Flask, request, jsonify
import joblib
import numpy as np # Import numpy untuk operasi array

warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')

# Pastikan MODEL_VERSION sesuai dengan versi terbaru dari script training (v2.3)
# Jika Anda melatih ulang model dengan versi yang berbeda, pastikan untuk memperbarui ini.
MODEL_VERSION = "v2.3" 

# Kita akan mencoba memuat model TF-IDF terlebih dahulu karena seringkali lebih akurat
# Jika Anda ingin menggunakan CountVectorizer, ubah nama filenya di sini.
model_filename = f'model_pengaduan_tfidf_{MODEL_VERSION}.pkl'
vectorizer_filename = f'vectorizer_pengaduan_tfidf_{MODEL_VERSION}.pkl'

model_path = os.path.join(MODEL_DIR, model_filename)
vectorizer_path = os.path.join(MODEL_DIR, vectorizer_filename)

model = None
vectorizer = None

try:
    print(f"Mencoba memuat model dari: {model_path}")
    print(f"Mencoba memuat vectorizer dari: {vectorizer_path}")
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    app.logger.info("Model dan Vectorizer berhasil dimuat.")
except Exception as e:
    print(f"Gagal memuat model atau vectorizer: {str(e)}")
    app.logger.error(f"Gagal memuat model atau vectorizer: {str(e)}")
    # Jika gagal memuat TF-IDF, coba memuat CountVectorizer sebagai fallback
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
    # Periksa apakah model dan vectorizer berhasil dimuat
    if model is None or vectorizer is None:
        return jsonify({'error': 'Inisialisasi model gagal. Silakan periksa log server.'}), 500
    
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'Input teks diperlukan.'}), 400
        
        # Transformasi teks input menggunakan vectorizer yang dimuat
        vec_text = vectorizer.transform([text])
        
        # Inisialisasi variabel untuk hasil prediksi
        prediction = 'Tidak Diketahui'
        prioritas_proba = 0.0
        reguler_proba = 0.0

        # Logika penanganan jika teks tidak memiliki kata kunci yang dikenali
        if vec_text.nnz == 0: # nnz (number of non-zero entries) untuk sparse matrix
            prediction = 'Reguler' # Default ke 'Reguler'
            prioritas_proba = 0.1 # Probabilitas rendah untuk Prioritas
            reguler_proba = 0.9  # Probabilitas tinggi untuk Reguler
            message = "Klasifikasi default: Tidak ada kata kunci yang dikenali dalam dataset training."
        else:
            # Lakukan prediksi
            prediction = model.predict(vec_text)[0]
            
            # Dapatkan probabilitas prediksi
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
            
            prioritas_proba = proba[prioritas_idx] if prioritas_idx != -1 else 0.0
            reguler_proba = proba[reguler_idx] if reguler_idx != -1 else 0.0
            message = "Prediksi berhasil."

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
    # Menggunakan port dari environment variable jika tersedia, default ke 5000
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))