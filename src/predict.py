import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
import torch.nn as nn

# ----------------------------
# Load model
# ----------------------------
def load_model(model_path="models/aeriscan_pneumonia.pth"):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model


# ----------------------------
# Predict single image
# ----------------------------
def predict_image(image_path):
    model = load_model()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)

    pneumonia_prob = probs[0][1].item()
    prediction = "PNEUMONIA" if pneumonia_prob > 0.5 else "NORMAL"

    return prediction, pneumonia_prob


if __name__ == "__main__":
    img = "sample_xray.png"  # replace with any test image
    pred, conf = predict_image(img)
    print(f"Prediction: {pred}")
    print(f"Confidence: {conf:.3f}")
