import os
import warnings
from flask import Flask, request, jsonify
import joblib
# from joblib import parallel_backend  # Jangan dulu pakai ini

warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')
MODEL_VERSION = "v2.1"

model_path = os.path.join(MODEL_DIR, f'model_pengaduan_{MODEL_VERSION}.pkl')
vectorizer_path = os.path.join(MODEL_DIR, f'vectorizer_pengaduan_{MODEL_VERSION}.pkl')

try:
    print(f"Model path: {model_path}")
    print(f"Vectorizer path: {vectorizer_path}")
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    app.logger.info("Model loaded successfully")
except Exception as e:
    print(f"Model loading failed: {str(e)}")
    app.logger.error(f"Model loading failed: {str(e)}")
    model = None
    vectorizer = None

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or vectorizer is None:
        return jsonify({'error': 'Model initialization failed'}), 500
    
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'Text is required'}), 400
        
        vec_text = vectorizer.transform([text])
        prediction = model.predict(vec_text)[0]
        
        return jsonify({
            'prediction': prediction,
            'model_version': MODEL_VERSION
        })
    
    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': 'Prediction failed'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
