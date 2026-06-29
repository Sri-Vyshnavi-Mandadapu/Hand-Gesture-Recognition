"""
utils.py
-----------
Utility functions used across the project.

Author: Sri Vyshnavi Mandadapu
"""

import os
import cv2
import csv
import time
import math
import numpy as np
from collections import deque, Counter


# =====================================================
# Folder Utilities
# =====================================================

def create_directories():
    """Create all required folders if they don't exist."""
    folders = [
        "dataset",
        "data",
        "models",
        "logs",
        "screenshots",
        "assets"
    ]

    for folder in folders:
        os.makedirs(folder, exist_ok=True)


# =====================================================
# Landmark Processing
# =====================================================

def normalize_landmarks(hand_landmarks):
    """
    Normalize landmarks using wrist as origin.
    """
    wrist = hand_landmarks.landmark[0]
    coords = []

    for lm in hand_landmarks.landmark:
        coords.extend([
            lm.x - wrist.x,
            lm.y - wrist.y,
            lm.z - wrist.z
        ])

    coords = np.array(coords, dtype=np.float32)
    max_value = np.max(np.abs(coords))

    if max_value > 0:
        coords /= max_value

    return coords


# =====================================================
# Feature Engineering
# =====================================================

def landmark_distance(lm1, lm2):
    return math.sqrt(
        (lm1.x - lm2.x) ** 2 +
        (lm1.y - lm2.y) ** 2 +
        (lm1.z - lm2.z) ** 2
    )


def landmark_angle(a, b, c):
    ba = np.array([
        a.x - b.x,
        a.y - b.y,
        a.z - b.z
    ])

    bc = np.array([
        c.x - b.x,
        c.y - b.y,
        c.z - b.z
    ])

    cosine = np.dot(ba, bc) / (
        np.linalg.norm(ba) *
        np.linalg.norm(bc) + 1e-6
    )

    cosine = np.clip(cosine, -1.0, 1.0)
    return np.arccos(cosine)


def extract_features(hand_landmarks):
    lm = hand_landmarks.landmark
    features = []

    # -----------------------------
    # 63 normalized coordinates
    # -----------------------------
    features.extend(
        normalize_landmarks(hand_landmarks)
    )

    # -----------------------------
    # Wrist → Fingertip distances
    # -----------------------------
    fingertips = [4, 8, 12, 16, 20]

    for tip in fingertips:
        features.append(
            landmark_distance(
                lm[0],
                lm[tip]
            )
        )

    # -----------------------------
    # Fingertip ↔ Fingertip distances
    # -----------------------------
    tip_pairs = [
        (4, 8),
        (4, 12),
        (4, 16),
        (4, 20),
        (8, 12),
        (8, 16),
        (8, 20),
        (12, 16),
        (12, 20),
        (16, 20)
    ]

    for p1, p2 in tip_pairs:
        features.append(
            landmark_distance(
                lm[p1],
                lm[p2]
            )
        )

    # -----------------------------
    # Finger Joint Angles
    # -----------------------------
    angle_triplets = [
        (0, 1, 2),
        (1, 2, 3),
        (2, 3, 4),
        (0, 5, 6),
        (5, 6, 7),
        (6, 7, 8),
        (0, 9, 10),
        (9, 10, 11),
        (10, 11, 12),
        (0, 13, 14),
        (13, 14, 15),
        (14, 15, 16),
        (0, 17, 18),
        (17, 18, 19),
        (18, 19, 20)
    ]

    for a, b, c in angle_triplets:
        features.append(
            landmark_angle(
                lm[a],
                lm[b],
                lm[c]
            )
        )

    return np.array(
        features,
        dtype=np.float32
    )


# =====================================================
# Bounding Box
# =====================================================

def get_bounding_box(hand_landmarks, image_shape, padding=20):
    h, w, _ = image_shape
    xs = []
    ys = []

    for lm in hand_landmarks.landmark:
        xs.append(int(lm.x * w))
        ys.append(int(lm.y * h))

    x1 = max(min(xs) - padding, 0)
    y1 = max(min(ys) - padding, 0)

    x2 = min(max(xs) + padding, w)
    y2 = min(max(ys) + padding, h)

    return x1, y1, x2, y2


# =====================================================
# Crop Hand
# =====================================================

def crop_hand(frame, bbox):
    x1, y1, x2, y2 = bbox
    return frame[y1:y2, x1:x2]


# =====================================================
# FPS
# =====================================================

class FPS:

    def __init__(self):
        self.previous = time.time()

    def update(self):
        current = time.time()
        fps = 1 / (current - self.previous)
        self.previous = current
        return int(fps)


# =====================================================
# Prediction History
# =====================================================

class PredictionHistory:

    def __init__(self, history_size=10):
        self.history = deque(maxlen=history_size)

    def update(self, prediction):
        self.history.append(prediction)
        return Counter(self.history).most_common(1)[0][0]


# =====================================================
# Screenshot
# =====================================================

def save_screenshot(frame, folder):
    filename = os.path.join(
        folder,
        f"{int(time.time())}.png"
    )
    cv2.imwrite(filename, frame)
    return filename


# =====================================================
# CSV Logger
# =====================================================

def log_prediction(log_file, hand, gesture, confidence):
    file_exists = os.path.isfile(log_file)

    with open(log_file, "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "Timestamp",
                "Hand",
                "Gesture",
                "Confidence"
            ])

        writer.writerow([
            time.strftime("%Y-%m-%d %H:%M:%S"),
            hand,
            gesture,
            round(confidence, 2)
        ])


# =====================================================
# Labels
# =====================================================

def load_labels(label_file):
    with open(label_file) as f:
        return [line.strip() for line in f]


# =====================================================
# Hand Label
# =====================================================

def get_hand_label(handedness):
    return handedness.classification[0].label