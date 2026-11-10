
import streamlit as st
import pandas as pd

from signbridge import config, dataset
from signbridge import train as trainer
from signbridge import video_import
from signbridge.model import model_exists
from signbridge.predictor import SignPredictor
from signbridge import speech

# Page configuration
st.set_page_config(
    page_title="SignBridge",
    page_icon="🤟",
    layout="wide"
)

# Header
st.title("🤟 SignBridge")
st.subheader("Indian Sign Language Recognition System")

st.write(
    "Real-time Indian Sign Language to text and speech. "
    "Collect samples, train your LSTM model and translate signs."
)

# Sample counts
def show_counts():
    counts = dataset.class_counts()
    if counts:
        st.dataframe(
            pd.DataFrame(
                [{"Sign": k, "Samples": v} for k, v in counts.items()]
            ),
            use_container_width=True
        )
    else:
        st.info("No training samples collected yet.")

# Navigation tabs
collect_tab, train_tab, live_tab, about_tab = st.tabs([
    "📷 Collect Data",
    "🧠 Train Model",
    "🤟 Live Translate",
    "ℹ️ About"
])

# 1. Collect data
with collect_tab:
    st.header("Collect Sign Samples")

    sign_name = st.text_input(
        "Enter sign name",
        placeholder="hello"
    )

    st.info(
        "Webcam sequence recording will be connected "
        "using Streamlit WebRTC."
    )

    st.subheader("Existing Training Samples")
    show_counts()

    if st.button("Import ISL Videos"):
        try:
            result = video_import.import_videos()
            st.success(
                f"Imported {result['imported']} videos."
            )
            show_counts()
        except Exception as e:
            st.error(str(e))

# 2. Train model
with train_tab:
    st.header("Train LSTM Model")

    epochs = st.slider(
        "Maximum epochs", 10, 300, 100, step=10
    )

    batch_size = st.slider(
        "Batch size", 4, 64, 16, step=4
    )

    if st.button("Start Training", type="primary"):
        try:
            with st.spinner("Training your LSTM model..."):
                report = trainer.train(
                    int(epochs),
                    int(batch_size)
                )

            st.success("Training completed!")

            st.metric(
                "Test Accuracy",
                f"{report['test_accuracy'] * 100:.2f}%"
            )

            st.json(report["per_label"])

        except Exception as e:
            st.error(f"Training failed: {e}")

# 3. Live translation
with live_tab:
    st.header("Live Sign Translation")

    if model_exists():
        st.success("Trained model found.")
    else:
        st.warning("Please train your model first.")

    st.info(
        "Live webcam sequence processing will be "
        "connected using Streamlit WebRTC."
    )

    confidence = st.slider(
        "Confidence threshold",
        0.30, 0.99,
        float(config.CONFIDENCE_THRESHOLD)
    )

    if st.button("Load Model"):
        try:
            st.session_state.predictor = SignPredictor(
                threshold=confidence,
                window=int(config.SMOOTHING_WINDOW),
                min_agreement=max(
                    1,
                    int(config.SMOOTHING_WINDOW * 0.6)
                )
            )
            st.success("Model loaded successfully!")
        except Exception as e:
            st.error(str(e))

# 4. About
with about_tab:
    st.header("About SignBridge")

    st.write("""
    SignBridge is a computer vision application
    designed to recognize Indian Sign Language.

    Technologies:
    - Python
    - Streamlit
    - MediaPipe Holistic
    - TensorFlow / LSTM
    - OpenCV
    - gTTS

    Features:
    - Sign data collection
    - LSTM model training
    - Live sign recognition
    - Text and speech output
    """)
