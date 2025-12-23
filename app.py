# app.py
# Aeriscan AI – Pneumonia Detection Streamlit App

import os
import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
from src.model import get_model

# ========================
# App Config
# ========================
st.set_page_config(
    page_title="Aeriscan AI",
    page_icon="🫁",
    layout="centered"
)

MODEL_PATH = "models/aeriscan_best.pth"

# ========================
# Load Model (cached)
# ========================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model file not found at: `{MODEL_PATH}`")
        st.stop()

    model = get_model()
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location="cpu")
    )
    model.eval()
    return model

model = load_model()

# ========================
# Image Transform
# ========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# ========================
# UI
# ========================
st.title("🫁 Aeriscan AI")
st.subheader("AI-powered Pneumonia Assessment from Chest X-rays")

st.markdown("""
Upload a chest X-ray image and Aeriscan AI will analyze it for **pneumonia-related patterns**.

⚠️ *This tool is for educational and research purposes only.*
""")

uploaded_file = st.file_uploader(
    "Upload Chest X-ray (JPG or PNG)",
    type=["jpg", "jpeg", "png"]
)

# ========================
# Prediction
# ========================
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="Uploaded X-ray", use_column_width=True)

    if st.button("Analyze X-ray"):
        with st.spinner("Analyzing image..."):
            input_tensor = transform(image).unsqueeze(0)

            with torch.no_grad():
                output = model(input_tensor)
                prob = torch.sigmoid(output).item()

            # Thresholds
            if prob >= 0.7:
                diagnosis = "Pneumonia Detected"
                risk = "High"
                color = "red"
            elif prob >= 0.4:
                diagnosis = "Uncertain Findings"
                risk = "Medium"
                color = "orange"
            else:
                diagnosis = "No Pneumonia Detected"
                risk = "Low"
                color = "green"

            st.markdown("---")
            st.subheader("🧪 AI Assessment")

            st.markdown(
                f"""
                **Diagnosis:** <span style="color:{color}; font-weight:bold;">{diagnosis}</span><br>
                **Confidence:** {prob * 100:.1f}%<br>
                **Risk Level:** {risk}
                """,
                unsafe_allow_html=True
            )

            st.progress(min(max(prob, 0.0), 1.0))

            st.markdown("""
            ---
            ⚠️ **Disclaimer:**  
            This AI model does **not** provide medical diagnoses.  
            Always consult a qualified healthcare professional.
            """)
