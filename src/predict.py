import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# ---------------- CONFIG ----------------
MODEL_PATH = "models/aeriscan_pneumonia.pth"
DEVICE = "cpu"
CLASS_NAMES = ["Normal", "Pneumonia"]

# ----------------------------------------

def load_model():
    # Build SAME model as training
    model = models.resnet18(weights=None)

    # 1-channel input (grayscale)
    model.conv1 = nn.Conv2d(
        1, 64, kernel_size=7, stride=2, padding=3, bias=False
    )

    # 2-class output
    model.fc = nn.Linear(model.fc.in_features, 2)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def predict_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)

    model = load_model()

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)

        confidence, pred_class = torch.max(probs, dim=1)

    diagnosis = CLASS_NAMES[pred_class.item()]
    confidence = confidence.item()

    # Risk interpretation
    p_pneumonia = probs[0][1].item()

    if p_pneumonia < 0.35:
        diagnosis = "Normal"
        confidence = 1 - p_pneumonia
        risk = "Low risk"

    elif p_pneumonia > 0.75:
        diagnosis = "Pneumonia"
        confidence = p_pneumonia
        risk = "High risk — seek medical evaluation"

    else:
        diagnosis = "Uncertain"
        confidence = p_pneumonia
        risk = "Indeterminate — clinical review recommended"


    return diagnosis, confidence, risk


if __name__ == "__main__":
    img_path = input("Enter path to chest X-ray image: ").strip()
    diagnosis, confidence, risk = predict_image(img_path)

    print("\n=== AERISCAN AI RESULT ===")
    print(f"Diagnosis : {diagnosis}")
    print(f"Confidence: {confidence:.2%}")
    print(f"Assessment: {risk}")
