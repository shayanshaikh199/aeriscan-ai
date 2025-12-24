from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io

from model import load_model, predict_pneumonia, risk_from_confidence

APP_NAME = "Aeriscan AI API"
MODEL_PATH = Path(__file__).parent / "models" / "aeriscan_best.pth"

app = FastAPI(title=APP_NAME)

# Allow your frontend dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load once at startup (fast predictions)
MODEL = None

@app.on_event("startup")
def _startup():
    global MODEL
    MODEL = load_model(MODEL_PATH)

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL is not None}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Upload an X-ray image (jpg/png). Returns JSON.
    Educational use only. Not a medical diagnosis.
    """
    if MODEL is None:
        return {"error": "Model not loaded"}

    raw = await file.read()
    try:
        img = Image.open(io.BytesIO(raw))
    except Exception:
        return {"error": "Invalid image file. Please upload a JPG/PNG image."}

    diagnosis, confidence = predict_pneumonia(MODEL, img)
    risk = risk_from_confidence(diagnosis, confidence)

    return {
        "diagnosis": diagnosis,
        "confidence": round(confidence, 4),   # 0..1
        "risk": risk,
        "disclaimer": "Educational use only. Not a medical diagnosis."
    }
