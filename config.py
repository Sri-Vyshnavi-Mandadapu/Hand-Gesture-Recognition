"""
Project Configuration File
Author: Sri Vyshnavi Mandadapu
"""

# ==========================
# Dataset
# ==========================

DATASET_PATH = "dataset"
DATA_PATH = "data"

# ==========================
# Model
# ==========================

MODEL_PATH = "models/hand_gesture_model.pkl"

# ==========================
# Labels
# ==========================

LABEL_FILE = "data/labels.txt"

# ==========================
# Logs
# ==========================

LOG_FILE = "logs/predictions.csv"

# ==========================
# Screenshot Folder
# ==========================

SCREENSHOT_PATH = "screenshots"

# ==========================
# Camera
# ==========================

CAMERA_ID = 0

FRAME_WIDTH = 1280

FRAME_HEIGHT = 720

# ==========================
# MediaPipe
# ==========================

MAX_HANDS = 2

DETECTION_CONFIDENCE = 0.7

TRACKING_CONFIDENCE = 0.7

# ==========================
# Prediction
# ==========================

HISTORY_SIZE = 10

CONFIDENCE_THRESHOLD = 0.70

# ==========================
# Machine Learning
# ==========================

RANDOM_STATE = 42

TEST_SIZE = 0.2

N_ESTIMATORS = 300

# ==========================
# Theme
# ==========================

APP_TITLE = "AI Hand Gesture Recognition"

AUTHOR = "Sri Vyshnavi Mandadapu"