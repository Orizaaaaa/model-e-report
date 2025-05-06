from http import HTTPStatus
import joblib
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np

# --- Inisialisasi Aplikasi ---
app = FastAPI(title="Model Pengaduan API")

# Enable CORS (untuk frontend di domain berbeda)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# --- Load Model & Vectorizer ---
try:
    model = joblib.load('model/model_pengaduan_v2.1.pkl')
    vectorizer = joblib.load('model/vectorizer_pengaduan_v2.1.pkl')
except Exception as e:
    raise RuntimeError(f"Gagal memuat model/vectorizer: {str(e)}") from e

# --- Schema Validasi Input ---
class PredictionRequest(BaseModel):
    text: str

# --- Endpoint Utama ---
@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        # Validasi input
        if not request.text.strip():
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST,
                detail="Input 'text' tidak boleh kosong"
            )

        # Transformasi teks
        vec_text = vectorizer.transform([request.text])
        
        # Prediksi
        prediction = model.predict(vec_text)[0]
        proba = model.predict_proba(vec_text)[0].tolist()

        # Mapping label untuk respons lebih deskriptif
        label_map = {
            0: {"label": "Prioritas", "deskripsi": "Laporan membutuhkan penanganan segera"},
            1: {"label": "Reguler", "deskripsi": "Laporan akan diproses dalam waktu 24 jam"}
        }

        return {
            "prediction": label_map[prediction]["label"],
            "confidence": round(max(proba), 2),
            "detail": label_map[prediction]["deskripsi"],
            "probabilities": {
                "Prioritas": proba[0],
                "Reguler": proba[1]
            }
        }

    except Exception as e:
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail=f"Terjadi kesalahan server: {str(e)}"
        )

# --- Health Check Endpoint ---
@app.get("/")
async def health_check():
    return {
        "status": "OK",
        "model_version": "v2.1",
        "ready": True,
        "endpoints": {
            "/predict": "POST dengan payload {'text': 'string'}"
        }
    }

# --- Konfigurasi untuk Vercel ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        timeout_keep_alive=120  # Untuk handle cold start
    )