from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import io
import os

from gradcam import GradCAM

app = FastAPI(title="Aeriscan AI Backend")

# -----------------------------
# CORS (frontend access)
# -----------------------------
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

# -----------------------------
# Paths (robust)
# -----------------------------
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BACKEND_DIR, ".."))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "aeriscan_best.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Load model once
# -----------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found at: {MODEL_PATH}\n"
        f"Make sure models/aeriscan_best.pth exists in project root."
    )

model = models.resnet18(weights=None)
model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
model.fc = torch.nn.Linear(model.fc.in_features, 2)

state = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state)
model.to(device)
model.eval()

# ✅ Grad-CAM uses the actual layer module (not a name)
gradcam = GradCAM(model, target_layer=model.layer4)

# -----------------------------
# Transform
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(1),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])


@app.get("/")
def root():
    return {"status": "ok", "message": "Aeriscan backend running"}


def _analyze_core(file: UploadFile):
    image_bytes = file.file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("L")

    image_tensor = transform(image).unsqueeze(0).to(device)

    # Forward pass (no_grad OK for probs)
    with torch.no_grad():
        output = model(image_tensor)
        probs = F.softmax(output, dim=1)

    normal_prob = float(probs[0, 0].item())
    pneumonia_prob = float(probs[0, 1].item())

    if pneumonia_prob >= 0.5:
        diagnosis = "Pneumonia"
        confidence = pneumonia_prob
        risk = "High" if pneumonia_prob >= 0.8 else "Moderate"
        class_idx = 1
    else:
        diagnosis = "Normal"
        confidence = normal_prob
        risk = "Low"
        class_idx = 0

    # Grad-CAM needs gradients, so generate without torch.no_grad()
    # We run it on the same image_tensor
    cam = gradcam.generate(image_tensor, class_idx=class_idx)
    heatmap_b64 = gradcam.to_base64_png(cam, out_size=(224, 224))

    return {
        "diagnosis": diagnosis,
        "confidence": confidence,
        "risk": risk,
        "heatmap": heatmap_b64
    }


# Swagger endpoint
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        result = _analyze_core(file)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})


# Frontend-friendly alias (so /api/analyze works too)
@app.post("/api/analyze")
async def analyze_api(file: UploadFile = File(...)):
    try:
        result = _analyze_core(file)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})
