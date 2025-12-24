from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights
from torchvision import transforms
from PIL import Image

DEVICE = "cpu"


def build_model() -> nn.Module:
    m = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    # change first conv to 1 channel (grayscale)
    m.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # 2 classes: Normal vs Pneumonia
    m.fc = nn.Linear(m.fc.in_features, 2)
    return m

# Same preprocessing for training/inference (safe default)
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # [0,1]
    transforms.Normalize(mean=[0.5], std=[0.5])  # grayscale normalize
])

def load_model(model_path: str | Path) -> nn.Module:
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at: {model_path.resolve()}")

    model = build_model().to(DEVICE)
    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model

@torch.inference_mode()
def predict_pneumonia(model: nn.Module, img: Image.Image) -> Tuple[str, float]:
    """
    Returns (diagnosis_label, confidence_probability)
    confidence is probability of predicted class.
    """
    # Force grayscale so input matches conv1(1 channel)
    img = img.convert("L")

    x = TRANSFORM(img).unsqueeze(0).to(DEVICE)  # [1,1,224,224]
    logits = model(x)  # [1,2]
    probs = torch.softmax(logits, dim=1)[0]  # [2]

    pred_idx = int(torch.argmax(probs).item())
    confidence = float(probs[pred_idx].item())

    # Convention: class 0 = Normal, class 1 = Pneumonia
    diagnosis = "Normal" if pred_idx == 0 else "Pneumonia"
    return diagnosis, confidence

def risk_from_confidence(diagnosis: str, conf: float) -> str:
    """
    Optional risk banding. This is NOT medical advice, just UI grouping.
    """
    if diagnosis == "Normal":
        if conf >= 0.85:
            return "Low"
        if conf >= 0.70:
            return "Moderate"
        return "Uncertain"
    else:
        if conf >= 0.85:
            return "High"
        if conf >= 0.70:
            return "Moderate"
        return "Uncertain"
