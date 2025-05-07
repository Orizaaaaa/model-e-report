import os
import warnings
from flask import Flask, request, jsonify
import joblib
from sklearn.externals.joblib import parallel_backend

# Suppress joblib warnings
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["JOBLIB_START_METHOD"] = "forkserver"
os.environ["LOKY_MAX_CPU_COUNT"] = "1"

app = Flask(__name__)

# Config paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
MODEL_VERSION = "v2.1"

# Model paths
model_path = os.path.join(MODEL_DIR, f'model_pengaduan_{MODEL_VERSION}.pkl')
vectorizer_path = os.path.join(MODEL_DIR, f'vectorizer_pengaduan_{MODEL_VERSION}.pkl')

# Load models with threading backend
try:
    with parallel_backend('threading'):
        model = joblib.load(
            model_path,
            mmap_mode='r',
            prefer='threads'
        )
        vectorizer = joblib.load(
            vectorizer_path,
            mmap_mode='r',
            prefer='threads'
        )
    app.logger.info("Model loaded successfully")
except Exception as e:
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
        
        # Process prediction
        with parallel_backend('threading'):
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