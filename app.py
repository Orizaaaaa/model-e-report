from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)

# Path absolut untuk model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_VERSION = "v2.1"

model = joblib.load(os.path.join(BASE_DIR, 'model', f'model_pengaduan_{MODEL_VERSION}.pkl'))
vectorizer = joblib.load(os.path.join(BASE_DIR, 'model', f'vectorizer_pengaduan_{MODEL_VERSION}.pkl'))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')
    vec_text = vectorizer.transform([text])
    prediction = model.predict(vec_text)[0]
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run()