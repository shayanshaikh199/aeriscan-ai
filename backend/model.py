import io
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

DEVICE = torch.device("cpu")

# IMPORTANT: your actual checkpoint name
# You said your file is models/aeriscan_best.pth
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "aeriscan_best.pth"

# ---- Model definition must match training ----
# Your checkpoint shows conv1 [64, 1, 7,7] AND fc [2,512]
# so: grayscale input (1 channel) + 2-class output
def build_model():
    model = models.resnet18(weights=None)

    # Make it 1-channel (grayscale)
    model.conv1 = nn.Conv2d(
        1, 64, kernel_size=7, stride=2, padding=3, bias=False
    )

    # 2 classes: Normal vs Pneumonia
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at: {MODEL_PATH}\n"
            f"Make sure models/aeriscan_best.pth exists."
        )

    model = build_model()
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model

# Transform: must match training expectations
TRANSFORM = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

def risk_from_confidence(conf_pct: float) -> str:
    # Simple risk mapping – tweak later if you want
    if conf_pct >= 85:
        return "High"
    if conf_pct >= 65:
        return "Medium"
    return "Low"

@torch.no_grad()
def predict_image_bytes(image_bytes: bytes, model):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    x = TRANSFORM(img).unsqueeze(0).to(DEVICE)  # [1,1,224,224]

    logits = model(x)  # [1,2]
    probs = torch.softmax(logits, dim=1)[0]  # [2]

    # class index 0 = Normal, 1 = Pneumonia
    conf, pred_idx = torch.max(probs, dim=0)

    pred_idx = int(pred_idx.item())
    conf_pct = float(conf.item() * 100.0)

    diagnosis = "Normal" if pred_idx == 0 else "Pneumonia"
    risk = risk_from_confidence(conf_pct)

    return diagnosis, round(conf_pct, 2), risk
