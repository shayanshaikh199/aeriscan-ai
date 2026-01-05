🩺 Aeriscan AI

Aeriscan AI is a full-stack medical imaging web application that analyzes chest X-ray images and predicts whether pneumonia is present.
The system provides a confidence score, risk level, and a visual explanation using Grad-CAM heatmaps to show which regions of the image most influenced the model’s prediction.

⚠️ Educational use only. Not a medical diagnosis.

✨ Features

Upload chest X-ray images through a clean, Apple-style UI

AI model predicts Pneumonia vs Normal

Confidence score with animated progress bar

Risk level classification (Low / Moderate / High)

Grad-CAM heatmap overlay for visual explainability

🔴 Red heatmap for pneumonia predictions

🟢 Green heatmap for normal predictions

Professional hover tooltip explaining heatmap meaning

FastAPI backend with interactive Swagger documentation

Fully responsive React + TypeScript frontend

🧠 How It Works

The user uploads a chest X-ray image

The image is sent to the FastAPI backend

A trained PyTorch CNN processes the image

The model:

Predicts the class (Pneumonia or Normal)

Calculates a confidence score

Generates a Grad-CAM heatmap

The frontend overlays the heatmap on the X-ray and displays the results in real time.

🛠️ Tech Stack
Backend

Python

PyTorch

FastAPI

Grad-CAM

Uvicorn

Frontend

React

TypeScript

Vite

CSS

▶️ Running the Project Locally
Backend Setup
cd backend
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
uvicorn main:app --reload


The backend will run at:
http://127.0.0.1:8000

Swagger docs available at:
http://127.0.0.1:8000/docs

Frontend Setup
cd frontend
npm install
npm run dev


The frontend will run at:
http://localhost:5173

🎨 Heatmap Explanation

🔴 Red overlay → Model detected features consistent with pneumonia

🟢 Green overlay → Model detected features consistent with normal lungs

Heatmaps highlight regions the model focused on — not exact pathology

Hover over the image to view an explanatory tooltip

👤 Author

Created by Shayan Shaikh
