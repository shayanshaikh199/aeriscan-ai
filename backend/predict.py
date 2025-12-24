import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os

# ================= CONFIG =================
MODEL_PATH = "models/aeriscan_best.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Thresholds (medical-style)
NORMAL_THRESHOLD = 0.35
PNEUMONIA_THRESHOLD = 0.65

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

# ================= TRANSFORM =================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# ================= LOAD MODEL =================
def load_model():
    model = models.resnet18(weights=None)

    # MUST MATCH TRAINING
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(512, 2)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    return model

# ================= PREDICT =================
def predict_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path).convert("L")
    image = transform(image).unsqueeze(0).to(DEVICE)

    model = load_model()

    with torch.no_grad():
        logits = model(image)
        probs = torch.softmax(logits, dim=1)[0]

    pneumonia_prob = probs[1].item()
    normal_prob = probs[0].item()

    # Medical-style decision logic
    if pneumonia_prob >= PNEUMONIA_THRESHOLD:
        diagnosis = "PNEUMONIA"
        risk = "HIGH"
    elif pneumonia_prob <= NORMAL_THRESHOLD:
        diagnosis = "NORMAL"
        risk = "LOW"
    else:
        diagnosis = "UNCERTAIN"
        risk = "MODERATE"

    confidence = pneumonia_prob if diagnosis != "NORMAL" else normal_prob

    return diagnosis, confidence, risk

# ================= CLI =================
if __name__ == "__main__":
    img_path = input("Enter path to chest X-ray image: ").strip()

    diagnosis, confidence, risk = predict_image(img_path)

    print("\n==== Aeriscan AI Diagnosis ====")
    print(f"Result:     {diagnosis}")
    print(f"Risk Level: {risk}")
    print(f"Confidence: {confidence * 100:.2f}%")
