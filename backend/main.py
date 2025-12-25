from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pathlib import Path
import torch
import torch.nn.functional as F
from torchvision import models, transforms

app = FastAPI()

# (Proxy makes CORS irrelevant, but leaving permissive CORS is fine for local dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Always resolve model path relative to project root
# Project root = .../aeriscan-ai
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "aeriscan_best.pth"

device = torch.device("cpu")

def build_model():
    m = models.resnet18(weights=None)
    m.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    m.fc = torch.nn.Linear(m.fc.in_features, 2)
    state = torch.load(MODEL_PATH, map_location=device)
    m.load_state_dict(state)
    m.eval()
    return m

model = build_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

@app.get("/health")
def health():
    return {"status": "ok"}

# ✅ IMPORTANT: expose /api/analyze so frontend fetch("/api/analyze") works
@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("L")
    x = transform(image).unsqueeze(0)

    with torch.no_grad():
        out = model(x)
        probs = F.softmax(out, dim=1)

    normal = float(probs[0, 0].item())
    pneumonia = float(probs[0, 1].item())

    if pneumonia >= 0.5:
        risk = "High" if pneumonia >= 0.80 else "Moderate"
        return {"diagnosis": "Pneumonia", "confidence": pneumonia, "risk": risk}

    return {"diagnosis": "Normal", "confidence": normal, "risk": "Low"}
