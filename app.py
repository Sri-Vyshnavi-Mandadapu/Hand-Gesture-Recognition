"""
app.py

AI Hand Gesture Recognition Dashboard
Author: Sri Vyshnavi Mandadapu
"""

import os
import json
import cv2
import av
import joblib
import numpy as np
import pandas as pd
import mediapipe as mp
import tensorflow as tf
import streamlit as st

from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase,
    RTCConfiguration
)

from config import (
    MODEL_PATH,
    SCALER_PATH,
    ENCODER_PATH,
    METRICS_PATH,
    LOG_PATH,
    SCREENSHOT_PATH,
    MAX_HANDS,
    DETECTION_CONFIDENCE,
    TRACKING_CONFIDENCE,
    CONFIDENCE_THRESHOLD,
    HISTORY_SIZE
)

from utils import (
    extract_features,
    get_bounding_box,
    PredictionHistory,
    log_prediction,
    get_hand_label
)

# =====================================================
# Streamlit Page
# =====================================================

st.set_page_config(
    page_title="AI Hand Gesture Recognition",
    page_icon="✋",
    layout="wide"
)

st.title("✋ AI Hand Gesture Recognition")

st.markdown("---")

# =====================================================
# Load Resources
# =====================================================

@st.cache_resource
def load_resources():
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    encoder = joblib.load(ENCODER_PATH)
    metrics = {}

    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            metrics = json.load(f)

    return model, scaler, encoder, metrics


model, scaler, encoder, metrics = load_resources()
st.write("Classes:", list(encoder.classes_))
st.write("Input Shape:", model.input_shape)
st.write("Output Shape:", model.output_shape)

# =====================================================
# Global Objects
# =====================================================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=MAX_HANDS,
    min_detection_confidence=DETECTION_CONFIDENCE,
    min_tracking_confidence=TRACKING_CONFIDENCE
)

left_history = PredictionHistory(HISTORY_SIZE)
right_history = PredictionHistory(HISTORY_SIZE)

log_file = os.path.join(
    LOG_PATH,
    "predictions.csv"
)

rtc_configuration = RTCConfiguration(
    {
        "iceServers": [
            {
                "urls": [
                    "stun:stun.l.google.com:19302"
                ]
            }
        ]
    }
)

# =====================================================
# Sidebar
# =====================================================

st.sidebar.header("⚙️ Settings")

confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    0.0,
    1.0,
    float(CONFIDENCE_THRESHOLD),
    0.05
)

show_landmarks = st.sidebar.checkbox(
    "Show Landmarks",
    value=True
)

show_bbox = st.sidebar.checkbox(
    "Show Bounding Box",
    value=True
)

show_confidence = st.sidebar.checkbox(
    "Show Confidence",
    value=True
)

# =====================================================
# Dashboard Metrics
# =====================================================

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "Classes",
        len(encoder.classes_)
    )

with c2:
    if metrics:
        st.metric(
            "Accuracy",
            f"{metrics['accuracy']*100:.2f}%"
        )
    else:
        st.metric(
            "Accuracy",
            "--"
        )

with c3:
    if metrics:
        st.metric(
            "Samples",
            metrics["samples"]
        )
    else:
        st.metric(
            "Samples",
            "--"
        )

with c4:
    if metrics:
        st.metric(
            "Features",
            metrics["features"]
        )
    else:
        st.metric(
            "Features",
            "--"
        )

st.markdown("---")

# =====================================================
# Webcam Section
# =====================================================

st.subheader("🎥 Live Gesture Recognition")

st.info(
    "Click 'START' below and allow camera access."
)

# =====================================================
# Screenshot Button
# =====================================================

take_screenshot = st.button(
    "📸 Capture Screenshot"
)

# =====================================================
# Video Processor
# =====================================================

class HandGestureProcessor(VideoProcessorBase):

    def __init__(self):
        self.model = model
        self.scaler = scaler
        self.encoder = encoder

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        image = cv2.flip(image, 1)
        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        results = hands.process(rgb)

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks,
                results.multi_handedness
            ):
                hand_name = get_hand_label(
                    handedness
                )

                if show_landmarks:
                    mp_draw.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )

                x1, y1, x2, y2 = get_bounding_box(
                    hand_landmarks,
                    image.shape
                )

                features = extract_features(
                    hand_landmarks
                )

                features = self.scaler.transform(
                    [features]
                )

                prediction = self.model.predict(
                    features,
                    verbose=0
                )

                confidence = float(
                    np.max(prediction)
                )

                class_id = int(
                    np.argmax(prediction)
                )

                gesture = self.encoder.inverse_transform(
                    [class_id]
                )[0]

                if hand_name == "Left":
                    gesture = left_history.update(
                        gesture
                    )
                else:
                    gesture = right_history.update(
                        gesture
                    )

                if confidence < confidence_threshold:
                    gesture = "Unknown"

                color = (0, 255, 0)
                if gesture == "Unknown":
                    color = (0, 0, 255)

                if show_bbox:
                    cv2.rectangle(
                        image,
                        (x1, y1),
                        (x2, y2),
                        color,
                        2
                    )

                cv2.putText(
                    image,
                    gesture,
                    (x1, y1 - 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2
                )

                cv2.putText(
                    image,
                    hand_name,
                    (x1, y1 - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2
                )

                if show_confidence:
                    cv2.putText(
                        image,
                        f"{confidence*100:.1f}%",
                        (x1, y2 + 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2
                    )

                if gesture != "Unknown":
                    log_prediction(
                        log_file,
                        hand_name,
                        gesture,
                        confidence
                    )

        if take_screenshot:
            os.makedirs(
                SCREENSHOT_PATH,
                exist_ok=True
            )

            filename = os.path.join(
                SCREENSHOT_PATH,
                f"screenshot_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.png"
            )

            cv2.imwrite(
                filename,
                image
            )

        return av.VideoFrame.from_ndarray(
            image,
            format="bgr24"
        )

# =====================================================
# Start Webcam
# =====================================================

webrtc_streamer(
    key="gesture-recognition",
    rtc_configuration=rtc_configuration,
    video_processor_factory=HandGestureProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_processing=True
)

st.markdown("---")

# =====================================================
# Prediction History
# =====================================================

st.subheader("Prediction History")

if os.path.exists(log_file):
    try:
        df = pd.read_csv(log_file)
        st.dataframe(
            df.tail(20),
            use_container_width=True,
            hide_index=True
        )
    except Exception:
        st.info("Prediction log is empty.")
# Added implicit else handling inside block fallback
else:
    st.info("No prediction logs found.")

# =====================================================
# Model Information
# =====================================================

st.markdown("---")

st.subheader("Model Information")

if metrics:
    c1, c2 = st.columns(2)

    with c1:
        st.metric(
            "Accuracy",
            f"{metrics['accuracy']*100:.2f}%"
        )

        st.metric(
            "Loss",
            f"{metrics['loss']:.4f}"
        )

    with c2:
        st.metric(
            "Samples",
            metrics["samples"]
        )

        st.metric(
            "Features",
            metrics["features"]
        )

# =====================================================
# Gesture Classes
# =====================================================

st.markdown("---")

st.subheader("Supported Gestures")

cols = st.columns(4)

for i, gesture in enumerate(encoder.classes_):
    cols[i % 4].success(gesture)

# =====================================================
# Training Graphs
# =====================================================

st.markdown("---")

st.subheader("Training Results")

col1, col2 = st.columns(2)

accuracy_path = os.path.join(
    "models",
    "accuracy.png"
)

loss_path = os.path.join(
    "models",
    "loss.png"
)

if os.path.exists(accuracy_path):
    with col1:
        st.image(
            accuracy_path,
            caption="Accuracy",
            use_container_width=True
        )

if os.path.exists(loss_path):
    with col2:
        st.image(
            loss_path,
            caption="Loss",
            use_container_width=True
        )

# =====================================================
# Confusion Matrix
# =====================================================

cm_path = os.path.join(
    "models",
    "confusion_matrix.png"
)

if os.path.exists(cm_path):
    st.markdown("---")
    st.subheader("Confusion Matrix")
    st.image(
        cm_path,
        use_container_width=True
    )

# =====================================================
# Screenshot Gallery
# =====================================================

st.markdown("---")

st.subheader("Screenshots")

os.makedirs(
    SCREENSHOT_PATH,
    exist_ok=True
)

images = sorted(
    [
        file
        for file in os.listdir(SCREENSHOT_PATH)
        if file.lower().endswith(
            (".png", ".jpg", ".jpeg")
        )
    ],
    reverse=True
)

if len(images) == 0:
    st.info("No screenshots available.")
else:
    cols = st.columns(3)
    for i, img in enumerate(images[:12]):
        cols[i % 3].image(
            os.path.join(
                SCREENSHOT_PATH,
                img
            ),
            caption=img,
            use_container_width=True
        )

# =====================================================
# Download Prediction Logs
# =====================================================

if os.path.exists(log_file):
    with open(log_file, "rb") as file:
        st.download_button(
            label="📥 Download Prediction Log",
            data=file,
            file_name="predictions.csv",
            mime="text/csv"
        )

# =====================================================
# Footer
# =====================================================

st.markdown("---")

st.markdown(
"""
## ✋ AI Hand Gesture Recognition

**Developed by:** Sri Vyshnavi Mandadapu

### Technologies

- TensorFlow
- MediaPipe
- OpenCV
- Streamlit
- Streamlit-WebRTC
- Scikit-Learn
- NumPy
- Pandas

---
Real-time AI-powered Hand Gesture Recognition using Deep Learning.
"""
)