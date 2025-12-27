import base64
import io
import numpy as np
import torch
from PIL import Image


class GradCAM:
    """
    Minimal Grad-CAM implementation.
    Works with models like ResNet18. You pass the target layer MODULE (not a name).
    Example: GradCAM(model, target_layer=model.layer4)
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        # Forward hook: save activations
        def forward_hook(_module, _inp, out):
            self.activations = out

        # Backward hook: save gradients
        # Use full backward hook to avoid partial-hook warnings
        def backward_hook(_module, _grad_input, grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        """
        Returns a normalized CAM (H,W) numpy array in [0,1].
        input_tensor: shape [1, C, H, W]
        """
        self.model.zero_grad(set_to_none=True)

        # Forward
        logits = self.model(input_tensor)

        # Pick class if not provided
        if class_idx is None:
            class_idx = int(torch.argmax(logits, dim=1).item())

        score = logits[0, class_idx]

        # Backward for that class score
        score.backward(retain_graph=True)

        # Safety checks
        if self.activations is None or self.gradients is None:
            raise RuntimeError("GradCAM hooks did not capture activations/gradients.")

        # Grad-CAM: global-average-pool gradients to get weights
        # activations: [1, K, H, W]
        # gradients:   [1, K, H, W]
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # [1, K, 1, 1]
        cam = (weights * self.activations).sum(dim=1, keepdim=False)  # [1, H, W]
        cam = torch.relu(cam)

        # Normalize to [0,1]
        cam = cam[0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.detach().cpu().numpy()

    @staticmethod
    def to_base64_png(cam: np.ndarray, out_size=(224, 224)) -> str:
        """
        Converts CAM [0,1] to a red/yellow heatmap PNG (transparent-ish) and returns base64 string.
        """
        cam = np.clip(cam, 0.0, 1.0)

        # Resize CAM to match model input size (frontend overlay expects same size as preview rendering)
        cam_img = Image.fromarray((cam * 255).astype(np.uint8), mode="L").resize(out_size, Image.BILINEAR)
        cam_arr = np.array(cam_img).astype(np.float32) / 255.0  # [H,W] in [0,1]

        # Simple "hot" colormap (no cv2 needed):
        # R = cam
        # G = cam^0.7 (brighter mid)
        # B = 0
        r = cam_arr
        g = np.power(cam_arr, 0.7)
        b = np.zeros_like(cam_arr)

        # Alpha controls transparency (stronger where activation is strong)
        a = np.clip(cam_arr * 0.75, 0.0, 0.75)

        rgba = np.stack([r, g, b, a], axis=-1)  # [H,W,4]
        rgba_u8 = (rgba * 255).astype(np.uint8)

        out = Image.fromarray(rgba_u8, mode="RGBA")

        buf = io.BytesIO()
        out.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
