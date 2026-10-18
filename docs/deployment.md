# Deployment

## Vercel frontend

Import the GitHub repository with root directory `frontend`. Choose Vite, build command `npm run build`, output `dist`, and Node.js 22 (at least 22.13). `frontend/vercel.json` adds basic response headers.

Set `VITE_API_URL=https://your-api.onrender.com` for the desired environment. Do not add `/api/analyze`. Only public configuration belongs in `VITE_` variables; these values are included in the browser bundle. Vite's local proxy is not part of the production build.

## Container API

`render.yaml` defines a Docker web service and uses `/ready` as its deployment health check. Set `ALLOWED_ORIGINS` when creating the Blueprint. Select a hosting plan after checking its current pricing and memory allowance. No paid plan is selected in the manifest.

The same container can run on another Docker host:

```bash
docker build -t aeriscan-api .
docker run --rm -p 8000:8000 -e ALLOWED_ORIGINS=http://localhost:5173 aeriscan-api
```

The container runs as a non-root user, installs CPU-only PyTorch, and uses one Uvicorn worker. Each additional worker loads another copy of the model. Training data and frontend dependencies are excluded from the Docker build context.

| Variable | Where | Default / meaning |
| --- | --- | --- |
| `VITE_API_URL` | Frontend build | Empty: use same-origin `/api`; local Vite proxies to port 8000 |
| `ALLOWED_ORIGINS` | Backend runtime | Local Vite origins; comma-separated exact frontend origins |
| `MODEL_PATH` | Backend runtime | `models/aeriscan_best.pth` at the project root |
| `PORT` | Container runtime | `8000`; host-assigned ports are supported |

The checkpoint is included for a self-contained demo and its checksum is in `models/SHA256SUMS`. A missing or incompatible model keeps `/ready` at HTTP 503 and logs the cause on the server. `/health` is a liveness endpoint and reports `model_ready`; it is not sufficient to prove inference is working.

## Smoke check

```bash
curl https://your-api.onrender.com/ready
curl -X POST https://your-api.onrender.com/api/analyze -F "file=@samples/normal1.jpeg"
```

Expect `diagnosis`, `confidence` (0–1), and a base64 PNG `heatmap`. Sample filenames are not verified ground-truth labels. Then test through the Vercel page to verify CORS and browser configuration.

## Troubleshooting

- **Frontend works locally but analysis fails after deployment:** configure `VITE_API_URL` and rebuild the frontend. The Vite proxy only runs in development.
- **CORS error:** add the exact frontend origin to `ALLOWED_ORIGINS` and redeploy the backend. Include preview origins individually when needed.
- **503 on `/ready`:** inspect backend startup logs and verify the checkpoint path and architecture. Do not substitute random weights.
- **First request is slow:** some hosts suspend idle services. The browser allows up to two minutes per analysis; use an always-on service if you need consistent latency.
- **Out of memory:** increase the backend memory allocation and keep one worker. Measure resident memory under concurrent uploads before choosing a plan.

The file-content read is bounded to 5 MB and decoded images to 20 million pixels (Pillow's warning threshold). Multipart parsing happens before the handler, so also configure an ingress request-size limit and rate limiting at your chosen host for an exposed service. Images may be temporarily spooled by the multipart parser; the upload handle is closed after each request. The app does not write a persistent upload directory or log image contents.

Removing datasets from the current tree does not remove them from existing Git history. Use a shallow clone for new checkouts. History cleanup is a separate operation because it changes commit IDs.

References: [Vite on Vercel](https://vercel.com/docs/frameworks/frontend/vite), [Render Blueprint specification](https://render.com/docs/blueprint-spec), [PyTorch CPU installation](https://pytorch.org/get-started/previous-versions/).
