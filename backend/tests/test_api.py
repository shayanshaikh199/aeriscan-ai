import base64
import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.main import MAX_UPLOAD_BYTES, app


def image_bytes(format="PNG"):
    stream = io.BytesIO()
    Image.new("L", (64, 48), 120).save(stream, format=format)
    return stream.getvalue()


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.pth"))
    with TestClient(app) as client:
        yield client


def test_missing_checkpoint_is_not_ready(client):
    assert client.get("/health").json() == {"status": "ok", "model_ready": False}
    assert client.get("/ready").status_code == 503
    assert client.post("/api/analyze", files={"file": ("scan.png", image_bytes(), "image/png")}).status_code == 503


@pytest.mark.parametrize("content,status", [(b"not an image", 415), (b"x" * (MAX_UPLOAD_BYTES + 1), 413), (image_bytes("GIF"), 415)], ids=["invalid-bytes", "oversized", "unsupported-format"])
def test_rejects_invalid_uploads(client, content, status):
    assert client.post("/api/analyze", files={"file": ("scan.png", content, "image/png")}).status_code == status


def test_missing_upload(client):
    assert client.post("/api/analyze").status_code == 422


def test_inference_failure_does_not_leak_details(client):
    class BrokenPredictor:
        def predict(self, image):
            raise RuntimeError("private checkpoint path")
    app.state.predictor = BrokenPredictor()
    response = client.post("/api/analyze", files={"file": ("scan.png", image_bytes(), "image/png")})
    assert response.status_code == 500
    assert "private" not in response.text


def test_cors(client):
    headers = {"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"}
    assert client.options("/api/analyze", headers=headers).headers["access-control-allow-origin"] == headers["Origin"]
    headers["Origin"] = "https://untrusted.example"
    assert "access-control-allow-origin" not in client.options("/api/analyze", headers=headers).headers


def test_checkpoint_prediction_and_heatmap(monkeypatch):
    checkpoint = Path(__file__).resolve().parents[2] / "models/aeriscan_best.pth"
    monkeypatch.setenv("MODEL_PATH", str(checkpoint))
    with TestClient(app) as client:
        assert client.get("/ready").status_code == 200
        response = client.post("/api/analyze", files={"file": ("scan.png", image_bytes(), "image/png")})
        assert response.status_code == 200
        data = response.json()
        assert data["diagnosis"] in {"Normal", "Pneumonia"}
        assert 0 <= data["confidence"] <= 1
        assert data["risk"] in {"Low", "Moderate", "High"}
        heatmap = Image.open(io.BytesIO(base64.b64decode(data["heatmap"])))
        assert heatmap.size == (224, 224)
        assert heatmap.mode == "RGBA"
