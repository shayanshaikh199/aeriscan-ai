import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

# ----------------------------------------
# Load trained pneumonia model (1-channel)
# ----------------------------------------
def load_model(model_path="models/aeriscan_pneumonia.pth"):
    model = models.resnet18(weights=None)

    # IMPORTANT: match training architecture
    model.conv1 = nn.Conv2d(
        in_channels=1,
        out_channels=64,
        kernel_size=7,
        stride=2,
        padding=3,
        bias=False
    )

    model.fc = nn.Linear(model.fc.in_features, 2)

    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)

    model.eval()
    return model


# ----------------------------------------
# Predict pneumonia from X-ray
# ----------------------------------------
def predict_image(image_path):
    model = load_model()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),  # 🔥 IMPORTANT
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    image = Image.open(image_path).convert("L")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)

    pneumonia_prob = probs[0, 1].item()
    prediction = "PNEUMONIA" if pneumonia_prob >= 0.5 else "NORMAL"

    return prediction, pneumonia_prob


# ----------------------------------------
# CLI test
# ----------------------------------------
if __name__ == "__main__":
    img_path = "sample2.jpg"  # make sure this exists
    pred, conf = predict_image(img_path)

    print(f"\n🩺 Aeriscan AI Result")
    print(f"Prediction : {pred}")
    print(f"Confidence : {conf:.3f}")
