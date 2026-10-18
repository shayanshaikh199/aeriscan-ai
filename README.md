# Aeriscan AI

[![CI](https://github.com/shayanshaikh199/aeriscan-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/shayanshaikh199/aeriscan-ai/actions/workflows/ci.yml)

Chest X-ray classification with a React interface, a FastAPI inference service, and Grad-CAM visualizations.

Upload a JPEG or PNG, run a grayscale ResNet18 model, and view a Grad-CAM activation overlay. The application returns a predicted class (`Normal` or `Pneumonia`) and its softmax score. The existing interface also displays a threshold-based risk label; this is a heuristic, not a clinical risk estimate.

Built by [Shayan Shaikh](https://github.com/shayanshaikh199).

## Run locally

Requires Python 3.13 and Node.js 22.13 or newer. Run commands from the repository root unless stated otherwise.

```bash
git clone --depth 1 https://github.com/shayanshaikh199/aeriscan-ai.git
cd aeriscan-ai
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
source .venv/bin/activate
```

Install and start the API:

```bash
# CPU wheels for Linux / Windows
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload
```

On macOS, install the same torch and torchvision versions from PyPI without `--index-url`.

In another terminal:

```bash
cd frontend
npm ci
npm run dev
```

Open [localhost:5173](http://localhost:5173). Vite forwards `/api` requests to port 8000 during development. The [API docs](http://127.0.0.1:8000/docs) describe the upload endpoint.

The included `models/aeriscan_best.pth` is the recovered project checkpoint. No model download, training run, or API key is required to start the app.

## Deploy

Deploy the **frontend to Vercel** and the **inference API as a Docker service**. The frontend is static; PyTorch and the checkpoint stay in the backend container.

1. Create a Render Blueprint from this repository using `render.yaml`. Set `ALLOWED_ORIGINS` to the exact Vercel origin, such as `https://your-project.vercel.app` (no path). Use a service with enough memory for PyTorch; 1 GB is a starting point, then measure actual usage.
2. Import this repository into Vercel. Set **Root Directory** to `frontend`, select **Vite**, and use Node.js 22 or newer. The checked-in configuration builds to `dist`.
3. Set `VITE_API_URL` in Vercel to the HTTPS origin of the Render API, with no `/api` suffix. Redeploy after changing it; Vite embeds this value at build time.
4. Check the API's `/ready` endpoint and upload an image on the deployed frontend. Add custom domains to `ALLOWED_ORIGINS` if used.

You can create the Vercel project first to reserve its URL, then deploy the backend and set `VITE_API_URL`. Preview deployments use different origins and must be explicitly allowed if you want them to call the API.

See [deployment notes](docs/deployment.md) for Docker commands, configuration, and troubleshooting. Hosting accounts and billing are configured separately; this repository does not provision them automatically.

## Architecture

```text
React + TypeScript (Vite / Vercel)
    â”‚ multipart POST /api/analyze
    â–¼
FastAPI (Docker / Render)
    â”‚ validate JPEG/PNG, decode, grayscale, resize to 224 Ã— 224
    â–¼
ResNet18 â†’ softmax class score
    â”‚ backward pass through layer4
    â–¼
Grad-CAM â†’ transparent PNG overlay â†’ browser
```

The checkpoint loads once at startup on CPU. A lock serializes inference because Grad-CAM hooks hold per-request activations and gradients. The API processes images in memory and does not create an image archive. Like any hosted request, uploads pass through the hosting providers.

| Path | Purpose |
| --- | --- |
| `frontend/src/` | Original upload flow, result display, and heatmap overlay |
| `backend/main.py` | Upload validation, CORS, health and analysis endpoints |
| `backend/model.py` | Architecture and preprocessing |
| `backend/inference.py` | Model lifecycle and serialized inference |
| `backend/gradcam.py` | Activation map generation and PNG encoding |
| `backend/train.py` | Training and validation loop |
| `models/` | Inference checkpoint and checksum |
| `samples/` | Original repository images for trying the upload flow |

## Validation

```bash
# From frontend/
npm run lint
npm run build

# From the root, with the Python environment activated
pip install -r backend/requirements-dev.txt
python -m pytest backend/tests -q
```

CI runs frontend lint/type checking/build and API tests, including a real checkpoint load and Grad-CAM output. These checks verify software behavior, not diagnostic accuracy.

## Model and data

Legacy DICOM/pneumothorax preparation scripts remain in the repository, but are not part of the JPEG pneumonia training or deployed API.

The network takes one grayscale channel, resizes to 224 Ã— 224, and normalizes with mean/std `0.5`. Output index `0` is `Normal`; index `1` is `Pneumonia`. The training script uses ImageFolder directories and saves the checkpoint with the lowest validation loss.

Training data is kept outside Git. To retrain, place the dataset in `backend/data/chest_xray/{train,val,test}/{NORMAL,PNEUMONIA}` and run:

```bash
pip install -r backend/requirements-train.txt
cd backend
python train.py
```

The training script downloads ImageNet weights for initialization. The original script saves weights under `backend/models/` when run from that directory. After reviewing a new checkpoint, copy it to the root `models/aeriscan_best.pth` for deployment. See the [model card](docs/model-card.md) for provenance gaps and evaluation limitations.

## Limitations

This is an educational project, not a clinically validated diagnostic tool. The softmax score is uncalibrated, and Grad-CAM highlights model influence rather than a verified disease location. The model does not check whether an upload is a chest X-ray. No held-out accuracy, sensitivity, or specificity is claimed because a reproducible evaluation report is not available.

The public API has a 5 MB per-file limit but no authentication or per-user rate limiter. Review hosting access controls before promoting a public demo. Use only de-identified images you are permitted to upload.
