"""HTTP API for the chest X-ray classification demo."""
from contextlib import asynccontextmanager
import io
import logging
import os
from pathlib import Path
import warnings

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 20_000_000


class Analysis(BaseModel):
    diagnosis: str
    confidence: float = Field(ge=0, le=1)
    heatmap: str
    risk: str


@asynccontextmanager
async def lifespan(app):
    app.state.predictor = None
    path = Path(os.getenv("MODEL_PATH", PROJECT_ROOT / "models/aeriscan_best.pth"))
    if path.is_file():
        try:
            from backend.inference import Predictor
            app.state.predictor = Predictor(path)
        except Exception:
            logger.exception("Unable to load the model checkpoint")
    else:
        logger.warning("Model checkpoint missing; inference is unavailable")
    yield


app = FastAPI(title="Aeriscan AI", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip().rstrip("/") for origin in os.getenv(
        "ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",") if origin.strip()],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/")
@app.get("/health")
def health():
    return {"status": "ok", "model_ready": app.state.predictor is not None}


@app.get("/ready")
def ready():
    if app.state.predictor is None:
        raise HTTPException(503, "The trained model is not available yet.")
    return {"status": "ready"}


@app.post("/analyze", response_model=Analysis, include_in_schema=False)
@app.post("/api/analyze", response_model=Analysis)
def analyze(file: UploadFile = File(...)):
    try:
        content = file.file.read(MAX_UPLOAD_BYTES + 1)
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "Choose an image smaller than 5 MB.")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(content)) as uploaded:
                    if uploaded.format not in {"JPEG", "PNG"}:
                        raise HTTPException(415, "Only JPEG and PNG images are supported.")
                    uploaded.load()
                    image = uploaded.convert("L")
        except (UnidentifiedImageError, OSError, ValueError):
            raise HTTPException(415, "This file is not a readable JPEG or PNG image.")
        except (Image.DecompressionBombError, Image.DecompressionBombWarning):
            raise HTTPException(413, "Image dimensions are too large.")
        if app.state.predictor is None:
            raise HTTPException(503, "The trained model is not available yet.")
        try:
            return app.state.predictor.predict(image)
        except Exception:
            logger.exception("Inference failed")
            raise HTTPException(500, "Analysis failed. Please try another image.")
    finally:
        file.file.close()
