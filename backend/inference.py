"""CPU inference with serialized access to Grad-CAM's mutable hooks."""
from threading import Lock
import torch
from backend.gradcam import GradCAM
from backend.model import TRANSFORM, build_model


class Predictor:
    def __init__(self, path):
        torch.set_num_threads(2)
        self.model = build_model()
        self.model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        self.model.eval()
        self.gradcam = GradCAM(self.model, self.model.layer4)
        self.lock = Lock()

    def predict(self, image):
        with self.lock:
            tensor = TRANSFORM(image).unsqueeze(0)
            with torch.no_grad():
                probabilities = torch.softmax(self.model(tensor), dim=1)[0]
            pneumonia_probability = float(probabilities[1].item())
            class_idx = 1 if pneumonia_probability >= 0.5 else 0
            risk = "Low" if class_idx == 0 else ("High" if pneumonia_probability >= 0.8 else "Moderate")
            cam = self.gradcam.generate(tensor, class_idx=class_idx)
            return {
                "diagnosis": ("Normal", "Pneumonia")[class_idx],
                "confidence": float(probabilities[class_idx].item()),
                "risk": risk,
                "heatmap": self.gradcam.to_base64_png(cam),
            }
