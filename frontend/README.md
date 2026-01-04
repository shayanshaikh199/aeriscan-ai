Aeriscan AI

AI-powered pneumonia detection from chest X-rays

Aeriscan AI is a full-stack medical imaging web application that analyzes chest X-ray images and predicts whether pneumonia is present. The system provides a confidence score, risk level, and a visual explanation using Grad-CAM heatmaps to highlight regions of the image the model focused on.

Features
-Upload chest X-ray images through a clean, Apple-style UI
-AI model predicts Pneumonia vs Normal
-Confidence score with animated progress bar
-Risk level classification (Low / High)
-Grad-CAM heatmap overlay for visual explainability
-Red heatmap for pneumonia, green for normal
-Professional hover tooltip explaining heatmap meaning
-FastAPI backend with Swagger documentation
-Fully responsive React + TypeScript frontend

How It Works 
-User uploads a chest X-ray image
-Image is sent to the FastAPI backend
-A trained PyTorch CNN model processes the image

The model:
-Predicts the class (Pneumonia or Normal)
-Calculates confidence
-Generates a Grad-CAM heatmap
-The frontend overlays the heatmap on the X-ray and displays results