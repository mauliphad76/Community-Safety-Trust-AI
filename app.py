from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

# ==========================================================
# COMMUNITY SAFETY & TRUST AI PLATFORM
# Main Dashboard
# ==========================================================

st.set_page_config(
    page_title="Community Safety & Trust AI",
    page_icon="🛡️",
    layout="wide"
)

# -----------------------------------------------------------
# CUSTOM CSS
# ----------------------------------------------------------

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 18px;
        color: #9aa4b2;
        margin-top: 0px;
    }

    .status {
        background-color: #10251b;
        border: 1px solid #1f7a4d;
        padding: 10px 16px;
        border-radius: 10px;
        color: #4ade80;
        font-weight: 600;
        text-align: center;
    }

    .module-card {
        padding: 22px;
        border-radius: 14px;
        border: 1px solid #30363d;
        background-color: #11161d;
        min-height: 210px;
    }

    .module-title {
        font-size: 23px;
        font-weight: 650;
    }

    .module-tech {
        font-size: 14px;
        color: #7dd3fc;
        font-weight: 600;
    }

    .module-description {
        color: #b8c0cc;
        font-size: 15px;
        line-height: 1.5;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ----------------------------------------------------------
# HEADER
# ----------------------------------------------------------

st.markdown(
    '<div class="main-title">🛡️ Community Safety & Trust AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered platform for safety, fraud detection and digital trust'
    '</div>',
    unsafe_allow_html=True
)

st.write("")

# Status
status_col1, status_col2 = st.columns([5, 1])

with status_col2:
    st.markdown(
        '<div class="status">● SYSTEM ONLINE</div>',
        unsafe_allow_html=True
    )

st.divider()

# ----------------------------------------------------------
# INTRO
# ----------------------------------------------------------

st.subheader("Community Safety Center")

st.write(
    "Choose an AI module below to analyze currency, "
    "visual activity, or suspicious digital content."
)

st.write("")

# ----------------------------------------------------------
# THREE MODULES
# ----------------------------------------------------------

col1, col2, col3 = st.columns(3)

# ---------- Currency ----------

with col1:

    st.markdown(
        """
        <div class="module-card">

        <div class="module-title">
        💵 Fake Currency Detector
        </div>

        <div class="module-tech">
        CNN • Computer Vision
        </div>

        <br>

        <div class="module-description">
        Analyze a currency image and predict whether
        the note is Real or Fake with a confidence score.
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    if st.button(
        "Open Currency Detector",
        use_container_width=True
    ):
        st.session_state["module"] = "currency"


# ---------- Activity ----------

with col2:

    st.markdown(
        """
        <div class="module-card">

        <div class="module-title">
        👁️ Suspicious Activity
        </div>

        <div class="module-tech">
        YOLO + Rule Engine • Computer Vision
        </div>

        <br>

        <div class="module-description">
        Detect people and objects from webcam/video
        and generate alerts using safety rules.
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    if st.button(
        "Open Activity Detector",
        use_container_width=True
    ):
        st.session_state["module"] = "activity"


# ---------- Scam ----------

with col3:

    st.markdown(
        """
        <div class="module-card">

        <div class="module-title">
        📰 Fake News / Scam Detector
        </div>

        <div class="module-tech">
        BERT + NLP • Text Analysis
        </div>

        <br>

        <div class="module-description">
        Analyze suspicious news or messages and
        predict Real/Fake with confidence.
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    if st.button(
        "Open Scam Detector",
        use_container_width=True
    ):
        st.session_state["module"] = "scam"


# ----------------------------------------------------------
# PLATFORM WORKFLOW
# ----------------------------------------------------------

st.divider()

st.subheader("How the Platform Works")

step1, step2, step3, step4 = st.columns(4)

with step1:
    st.markdown("### 01")
    st.write("**Input**")
    st.caption("Image • Video • Text")

with step2:
    st.markdown("### 02")
    st.write("**AI Analysis**")
    st.caption("CNN • YOLO • BERT")

with step3:
    st.markdown("### 03")
    st.write("**Decision**")
    st.caption("Prediction + Confidence")

with step4:
    st.markdown("### 04")
    st.write("**Action**")
    st.caption("Result • Warning • Alert")


# ----------------------------------------------------------
# FOOTER
# ----------------------------------------------------------

st.divider()

st.caption(
    "Community Safety & Trust AI Platform | "
    "AI-assisted decision support system"
)

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "currency_cnn_best.keras"


@st.cache_resource
def load_currency_model():
    import tensorflow as tf

    return tf.keras.models.load_model(MODEL_PATH, compile=False)


def preprocess_currency_image(uploaded_file):
    uploaded_file.seek(0)
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((224, 224), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32)[None, ...] / 255.0


def render_currency_detector():
    st.subheader("Currency Detector")
    st.write("Upload a currency image for an AI-based Real or Fake prediction.")

    if st.button("Back to Dashboard", key="currency_back"):
        st.session_state["module"] = None
        st.rerun()

    uploaded_file = st.file_uploader(
        "Upload currency image",
        type=["jpg", "jpeg", "png", "avif", "webp"],
        key="currency_upload",
    )

    if uploaded_file is None:
        st.info("Upload an image to begin analysis.")
        return

    try:
        uploaded_file.seek(0)
        display_image = Image.open(uploaded_file).convert("RGB")
        st.image(display_image, caption="Uploaded currency image", use_container_width=True)
        model = load_currency_model()
        image_array = preprocess_currency_image(uploaded_file)
        fake_probability = float(model.predict(image_array, verbose=0)[0][0])
        predicted_label = 1 if fake_probability >= 0.5 else 0
        confidence = fake_probability if predicted_label == 1 else 1.0 - fake_probability

        if confidence < 0.60:
            st.warning("UNCLEAR — Please scan the note again.")
        else:
            prediction = "FAKE" if predicted_label == 1 else "REAL"
            st.success(f"Prediction: {prediction}")
            st.metric("Confidence", f"{confidence * 100:.2f}%")

        st.caption(
            "This AI result is an image-based prediction and should not be treated "
            "as a definitive bank-grade counterfeit verification."
        )
    except Exception as error:
        st.error(f"Unable to analyze this image: {error}")


if st.session_state.get("module") == "currency":
    st.divider()
    render_currency_detector()