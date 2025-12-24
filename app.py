import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import time

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Aeriscan AI",
    layout="centered"
)

# =========================
# APPLE-STYLE CSS
# =========================
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont,
                 "SF Pro Display", "SF Pro Text",
                 "Helvetica Neue", Helvetica, Arial, sans-serif;
    background-color: #ffffff;
    color: #111111;
}

/* Hide Streamlit spinner */
.stSpinner {
    display: none !important;
}

h1 {
    font-weight: 600;
    letter-spacing: -0.02em;
}

.caption {
    color: #6e6e73;
    font-size: 15px;
    margin-bottom: 30px;
}

/* Loading card */
.loading-card {
    background: #f5f5f7;
    border-radius: 18px;
    padding: 48px;
    margin-top: 30px;
    text-align: center;
    border: 1px solid #e5e5ea;
}

.loading-title {
    font-size: 18px;
    font-weight: 500;
    margin-bottom: 22px;
}

/* Apple-style progress bar */
.loading-bar-bg {
    width: 100%;
    height: 6px;
    background: #e5e5ea;
    border-radius: 999px;
    overflow: hidden;
}

.loading-bar {
    height: 100%;
    width: 40%;
    background: linear-gradient(
        90deg,
        #6366f1,
        #8b5cf6,
        #6366f1
    );
    animation: slide 1.4s ease-in-out infinite;
}

@keyframes slide {
    0% { transform: translateX(-100%); }
    50% { transform: translateX(120%); }
    100% { transform: translateX(120%); }
}

/* Result card */
.result-card {
    background: #f5f5f7;
    border-radius: 18px;
    padding: 36px;
    margin-top: 30px;
    border: 1px solid #e5e5ea;
    animation: fadeIn 0.5s ease;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(6px); }
    to { opacity: 1; transform: translateY(0); }
}

.diagnosis {
    font-size: 22px;
    font-weight: 600;
    margin-bottom: 24px;
}

.normal { color: #1d7f46; }
.pneumonia { color: #b91c1c; }

.metric-label {
    font-size: 13px;
    color: #6e6e73;
    margin-top: 20px;
}

.metric-value {
    font-size: 26px;
    font-weight: 500;
    margin-top: 6px;
}

.bar-bg {
    width: 100%;
    height: 18px;
    background: #e5e5ea;
    border-radius: 999px;
    margin-top: 18px;
    overflow: hidden;
}

.bar-fill {
    height: 100%;
    border-radius: 999px;
}

.bar-normal {
    background: linear-gradient(90deg, #22c55e, #4ade80);
}

.bar-pneumonia {
    background: linear-gradient(90deg, #ef4444, #f87171);
}

.disclaimer {
    margin-top: 24px;
    font-size: 12px;
    color: #9ca3af;
}
</style>
""", unsafe_allow_html=True)

# =========================
# MODEL
# =========================
MODEL_PATH = "models/aeriscan_best.pth"

@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    model.conv1 = torch.nn.Conv2d(1, 64, 7, 2, 3, bias=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(1),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def predict_image(image):
    model = load_model()
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        probs = F.softmax(output, dim=1)

    normal = probs[0, 0].item()
    pneumonia = probs[0, 1].item()

    if pneumonia > 0.5:
        return "Pneumonia", pneumonia, "High" if pneumonia > 0.8 else "Moderate"
    else:
        return "Normal", normal, "Low"

# =========================
# UI
# =========================
st.title("Aeriscan AI")
st.markdown('<div class="caption">AI-powered chest X-ray analysis</div>', unsafe_allow_html=True)

uploaded = st.file_uploader("Upload a chest X-ray image", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded).convert("L")
    st.image(image, use_container_width=True)

    if st.button("Analyze", use_container_width=True):

        # ---------- Loading UI ----------
        loading = st.empty()
        loading.markdown("""
        <div class="loading-card">
            <div class="loading-title">Analyzing X-ray</div>
            <div class="loading-bar-bg">
                <div class="loading-bar"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Simulated processing delay (feels intentional)
        time.sleep(1.2)

        diagnosis, confidence, risk = predict_image(image)

        loading.empty()

        bar_width = int(confidence * 100)
        bar_class = "bar-normal" if diagnosis == "Normal" else "bar-pneumonia"
        diag_class = "normal" if diagnosis == "Normal" else "pneumonia"

        html = f"""
        <div class="result-card">
            <div class="diagnosis {diag_class}">
                Diagnosis: {diagnosis}
            </div>

            <div class="metric-label">Confidence</div>
            <div class="metric-value">{confidence*100:.2f}%</div>

            <div class="bar-bg">
                <div class="bar-fill {bar_class}" style="width:{bar_width}%"></div>
            </div>

            <div class="metric-label">Risk Level</div>
            <div class="metric-value">{risk}</div>

            <div class="disclaimer">
                Educational use only. Not a medical diagnosis.
            </div>
        </div>
        """

        st.markdown(html, unsafe_allow_html=True)
