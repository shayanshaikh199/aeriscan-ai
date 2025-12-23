import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

# =========================
# CONFIG
# =========================
MODEL_PATH = "models/aeriscan_best.pth"
DEVICE = "cpu"

st.set_page_config(
    page_title="Aeriscan AI",
    page_icon="🫁",
    layout="centered"
)

# =========================
# LOAD MODEL (MATCH TRAINING)
# =========================
@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)

    # 🔴 CRITICAL: grayscale model (1 channel)
    model.conv1 = torch.nn.Conv2d(
        1, 64, kernel_size=7, stride=2, padding=3, bias=False
    )

    # Binary classifier (Normal vs Pneumonia)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)

    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)

    model.eval()
    model.to(DEVICE)

    return model


# =========================
# IMAGE TRANSFORM (MATCH TRAINING)
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),  # 🔴 MUST be 1
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


# =========================
# PREDICTION
# =========================
def predict_image(image: Image.Image):
    model = load_model()

    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(image)            # shape: [1, 2]
        probs = F.softmax(output, dim=1)

        normal_prob = probs[0, 0].item()
        pneumonia_prob = probs[0, 1].item()

    if pneumonia_prob >= 0.5:
        diagnosis = "Pneumonia"
        confidence = pneumonia_prob
        risk = "High" if confidence > 0.8 else "Moderate"
    else:
        diagnosis = "Normal"
        confidence = normal_prob
        risk = "Low"

    return diagnosis, confidence, risk


# =========================
# STREAMLIT UI
# =========================
st.title("🫁 Aeriscan AI")
st.subheader("AI-Powered Chest X-Ray Pneumonia Detection")

uploaded_file = st.file_uploader(
    "Upload a chest X-ray image (JPG / PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="Uploaded Chest X-ray", use_container_width=True)

    if st.button("Analyze"):
        with st.spinner("Analyzing X-ray..."):
            diagnosis, confidence, risk = predict_image(image)

        st.markdown("---")
        st.subheader("🩺 Result")

        if diagnosis == "Pneumonia":
            st.error(f"**Diagnosis:** {diagnosis}")
        else:
            st.success(f"**Diagnosis:** {diagnosis}")

        st.write(f"**Confidence:** {confidence * 100:.2f}%")
        st.write(f"**Risk Level:** {risk}")

        st.progress(confidence)

        st.caption(
            "⚠️ Educational use only. Not a medical diagnosis."
        )
