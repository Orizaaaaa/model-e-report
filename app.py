from flask import Flask, request, jsonify
import joblib
import os
from platformdirs import user_data_dir  # Modifikasi penting

app = Flask(__name__)

# Konfigurasi path cross-platform
APP_NAME = "model-e-report"
MODEL_VERSION = "v2.1"

# Buat direktori model jika belum ada
model_dir = os.path.join(user_data_dir(APP_NAME), "models")
os.makedirs(model_dir, exist_ok=True)

# Path model
model_path = os.path.join(model_dir, f'model_pengaduan_{MODEL_VERSION}.pkl')
vectorizer_path = os.path.join(model_dir, f'vectorizer_pengaduan_{MODEL_VERSION}.pkl')

# Load model
try:
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
except Exception as e:
    raise RuntimeError(f"Gagal load model: {str(e)}")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'Text tidak boleh kosong'}), 400
    
    try:
        vec_text = vectorizer.transform([text])
        prediction = model.predict(vec_text)[0]
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))