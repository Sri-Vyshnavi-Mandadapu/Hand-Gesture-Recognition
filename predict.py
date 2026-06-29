"""
predict.py

Real-Time Hand Gesture Recognition
TensorFlow + MediaPipe

Author: Sri Vyshnavi Mandadapu
"""

import os
import cv2
import joblib
import numpy as np
import mediapipe as mp
import tensorflow as tf

from config import (
    MODEL_PATH,
    SCALER_PATH,
    ENCODER_PATH,
    LABEL_FILE,
    LOG_PATH,
    SCREENSHOT_PATH,
    CAMERA_ID,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    MAX_HANDS,
    DETECTION_CONFIDENCE,
    TRACKING_CONFIDENCE,
    CONFIDENCE_THRESHOLD,
    HISTORY_SIZE
)

from utils import (
    extract_features,
    get_bounding_box,
    FPS,
    PredictionHistory,
    save_screenshot,
    log_prediction,
    get_hand_label
)

# ============================================================
# Create folders
# ============================================================
os.makedirs(LOG_PATH, exist_ok=True)
os.makedirs(SCREENSHOT_PATH, exist_ok=True)

# ============================================================
# Load Model
# ============================================================
print("=" * 60)
print("Loading TensorFlow Model...")
print("=" * 60)

model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
label_encoder = joblib.load(ENCODER_PATH)

print("Model Loaded Successfully!")

# ============================================================
# MediaPipe
# ============================================================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=MAX_HANDS,
    min_detection_confidence=DETECTION_CONFIDENCE,
    min_tracking_confidence=TRACKING_CONFIDENCE
)

# ============================================================
# Camera
# ============================================================
cap = cv2.VideoCapture(CAMERA_ID)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

if not cap.isOpened():
    print("Cannot open camera.")
    exit()

# ============================================================
# Utilities
# ============================================================
fps = FPS()
history = PredictionHistory(HISTORY_SIZE)
log_file = os.path.join(LOG_PATH, "predictions.csv")

print("\nPress Q -> Quit")
print("Press C -> Screenshot\n")

# ============================================================
# Main Loop
# ============================================================
while True:
    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = hands.process(rgb)
    current_fps = fps.update()

    if (
        results.multi_hand_landmarks
        and results.multi_handedness
    ):
        for hand_landmarks, handedness in zip(
            results.multi_hand_landmarks,
            results.multi_handedness
        ):
            # Draw landmarks
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Bounding Box
            x1, y1, x2, y2 = get_bounding_box(
                hand_landmarks,
                frame.shape
            )

            # Hand Label
            hand_name = get_hand_label(
                handedness
            )

            # Normalize Landmarks & Extract Engineered Features
            features = extract_features(
                hand_landmarks
            )

            # Scale Features
            features = scaler.transform(
                [features]
            )

            # ============================================================
            # Prediction
            # ============================================================
            prediction = model.predict(
                features,
                verbose=0
            )

            confidence = float(np.max(prediction))
            predicted_class = int(np.argmax(prediction))

            gesture = label_encoder.inverse_transform(
                [predicted_class]
            )[0]

            # ============================================================
            # Prediction Smoothing
            # ============================================================
            gesture = history.update(gesture)

            # ============================================================
            # Ignore Low Confidence
            # ============================================================
            if confidence < CONFIDENCE_THRESHOLD:
                gesture = "Unknown"

            # ============================================================
            # Draw Bounding Box
            # ============================================================
            color = (0, 255, 0)
            if gesture == "Unknown":
                color = (0, 0, 255)

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            # ============================================================
            # Gesture Text
            # ============================================================
            cv2.putText(
                frame,
                f"{gesture}",
                (x1, y1 - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                2
            )

            cv2.putText(
                frame,
                f"{hand_name}",
                (x1, y1 - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"{confidence*100:.1f}%",
                (x1, y2 + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            # ============================================================
            # CSV Logging
            # ============================================================
            if gesture != "Unknown":
                log_prediction(
                    log_file,
                    hand_name,
                    gesture,
                    confidence
                )

    # ============================================================
    # FPS Display
    # ============================================================
    cv2.putText(
        frame,
        f"FPS : {current_fps}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Q : Quit",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "C : Screenshot",
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "AI Hand Gesture Recognition",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    # Screenshot
    if key == ord("c"):
        filename = save_screenshot(
            frame,
            SCREENSHOT_PATH
        )
        print(f"Screenshot saved: {filename}")

    # Quit
    if key == ord("q"):
        break

# ============================================================
# Cleanup
# ============================================================
print("\nClosing application...")

cap.release()
hands.close()
cv2.destroyAllWindows()

print("Application closed successfully.")