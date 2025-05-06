# app.py
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load model dan vectorizer
MODEL_VERSION = "v2.1"
model = joblib.load(f'model_pengaduan_{MODEL_VERSION}.pkl')
vectorizer = joblib.load(f'vectorizer_pengaduan_{MODEL_VERSION}.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data['text']
    vec_text = vectorizer.transform([text])
    prediction = model.predict(vec_text)[0]
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run()