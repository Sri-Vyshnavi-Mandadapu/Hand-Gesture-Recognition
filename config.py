"""
Project Configuration
"""

import os

# ==========================================================
# Paths
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(BASE_DIR, "dataset")
LEAP_DATASET = os.path.join(DATASET_PATH, "LeapGestRecog")
CUSTOM_DATASET = os.path.join(DATASET_PATH, "Custom")

DATA_PATH = os.path.join(BASE_DIR, "data")
MODELS_PATH = os.path.join(BASE_DIR, "models")
LOG_PATH = os.path.join(BASE_DIR, "logs")
SCREENSHOT_PATH = os.path.join(BASE_DIR, "screenshots")

CSV_PATH = os.path.join(DATA_PATH, "data.csv")
LABEL_FILE = os.path.join(DATA_PATH, "labels.txt")

MODEL_PATH = os.path.join(MODELS_PATH, "gesture_model.keras")
SCALER_PATH = os.path.join(DATA_PATH, "scaler.pkl")
ENCODER_PATH = os.path.join(DATA_PATH, "label_encoder.pkl")

METRICS_PATH = os.path.join(MODELS_PATH, "metrics.json")

# ==========================================================
# Camera
# ==========================================================

CAMERA_ID = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# ==========================================================
# MediaPipe
# ==========================================================

MAX_HANDS = 2
DETECTION_CONFIDENCE = 0.7
TRACKING_CONFIDENCE = 0.7

# ==========================================================
# Training
# ==========================================================

TEST_SIZE = 0.2
RANDOM_STATE = 42

BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.001

# ==========================================================
# Prediction
# ==========================================================

CONFIDENCE_THRESHOLD = 0.45
HISTORY_SIZE = 10