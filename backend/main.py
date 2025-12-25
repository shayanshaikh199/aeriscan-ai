from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from model import load_model, predict_image_bytes

app = FastAPI(title="Aeriscan API")

# Allow your React frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load once at startup (fast)
MODEL = load_model()

@app.get("/")
def root():
    return {"status": "ok", "message": "Aeriscan backend running. Use POST /analyze"}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """
    Accepts an uploaded image (jpg/png) and returns JSON:
    { diagnosis: "Normal"|"Pneumonia", confidence: 0-100, risk: "Low"|"Medium"|"High" }
    """
    image_bytes = await file.read()
    diagnosis, confidence_pct, risk = predict_image_bytes(image_bytes, MODEL)

    return {
        "diagnosis": diagnosis,
        "confidence": confidence_pct,
        "risk": risk
    }
