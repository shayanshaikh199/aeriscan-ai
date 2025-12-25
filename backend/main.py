from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import io
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Aeriscan AI API")



app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# PATHS
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "aeriscan_best.pth"

device = torch.device("cpu")
model = None  # <-- IMPORTANT

# =========================
# LOAD MODEL (LAZY)
# =========================
def get_model():
    global model
    if model is None:
        m = models.resnet18(weights=None)
        m.conv1 = torch.nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        m.fc = torch.nn.Linear(m.fc.in_features, 2)
        m.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        m.eval()
        model = m
    return model

# =========================
# TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(1),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# =========================
# ROUTES
# =========================
@app.get("/")
def root():
    return {"status": "Aeriscan backend running"}

@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("L")
    image = transform(image).unsqueeze(0)

    model = get_model()

    with torch.no_grad():
        output = model(image)
        probs = F.softmax(output, dim=1)

    normal = probs[0, 0].item()
    pneumonia = probs[0, 1].item()

    if pneumonia > 0.5:
        diagnosis = "Pneumonia"
        confidence = pneumonia
        risk = "High" if pneumonia > 0.8 else "Moderate"
    else:
        diagnosis = "Normal"
        confidence = normal
        risk = "Low"

    return {
        "diagnosis": diagnosis,
        "confidence": round(confidence, 4),
        "risk": risk
    }
