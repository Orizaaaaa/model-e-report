# app.py
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os  # Tambahkan ini

app = Flask(__name__)

# Path dinamis untuk model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_VERSION = "v2.1"

# Load model dan vectorizer
model = joblib.load(os.path.join(BASE_DIR, f'model/model_pengaduan_{MODEL_VERSION}.pkl'))
vectorizer = joblib.load(os.path.join(BASE_DIR, f'model/vectorizer_pengaduan_{MODEL_VERSION}.pkl'))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data['text']
    vec_text = vectorizer.transform([text])
    prediction = model.predict(vec_text)[0]
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run()