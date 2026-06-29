"""
create_dataset.py

Creates a landmark dataset from LeapGestRecog using MediaPipe Hands.

Output:
    data/data.csv
    data/labels.txt
"""

import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import extract_features
from config import (
    LEAP_DATASET,
    CSV_PATH,
    LABEL_FILE,
    DATA_PATH,
    DETECTION_CONFIDENCE,
    TRACKING_CONFIDENCE
)

# -------------------------------------------------------
# Create output folder
# -------------------------------------------------------
os.makedirs(DATA_PATH, exist_ok=True)

# -------------------------------------------------------
# MediaPipe
# -------------------------------------------------------
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=DETECTION_CONFIDENCE,
    min_tracking_confidence=TRACKING_CONFIDENCE
)

# -------------------------------------------------------
# Gesture Mapping
# -------------------------------------------------------
GESTURE_MAP = {
    "01_palm": "Palm",
    "02_l": "L",
    "03_fist": "Fist",
    "04_fist_moved": "FistMoved",
    "05_thumb": "Thumb",
    "06_index": "Index",
    "07_ok": "OK",
    "08_palm_moved": "PalmMoved",
    "09_c": "C",
    "10_down": "Down"
}

# -------------------------------------------------------
# Storage
# -------------------------------------------------------
dataset = []
labels = set()
processed = 0
skipped = 0

print("=" * 60)
print("Building Landmark Dataset")
print("=" * 60)

subjects = sorted(os.listdir(LEAP_DATASET))

# -------------------------------------------------------
# Read Dataset
# -------------------------------------------------------
for subject in subjects:
    subject_path = os.path.join(LEAP_DATASET, subject)

    if not os.path.isdir(subject_path):
        continue

    print(f"\nProcessing Subject : {subject}")
    gesture_folders = sorted(os.listdir(subject_path))

    for folder in gesture_folders:
        folder_path = os.path.join(subject_path, folder)

        if not os.path.isdir(folder_path):
            continue

        if folder not in GESTURE_MAP:
            continue

        label = GESTURE_MAP[folder]
        labels.add(label)

        image_files = [
            f for f in os.listdir(folder_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        for image_name in tqdm(
            image_files,
            desc=label,
            leave=False
        ):
            image_path = os.path.join(folder_path, image_name)
            image = cv2.imread(image_path)

            if image is None:
                skipped += 1
                continue

            rgb = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )

            results = hands.process(rgb)

            if not results.multi_hand_landmarks:
                skipped += 1
                continue

            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Using imported extract_features function to get the complete landmark list
            features = extract_features(hand_landmarks).tolist()

            row = features + [label]
            dataset.append(row)
            processed += 1

print("\nFinished extracting landmarks.")

# -------------------------------------------------------
# Save Dataset
# -------------------------------------------------------
if len(dataset) == 0:
    print("\nNo valid hand landmarks were extracted.")
    hands.close()
    exit()

columns = []

# 63 normalized coordinates
for i in range(21):
    columns.extend([
        f"x{i}",
        f"y{i}",
        f"z{i}"
    ])

# 5 wrist-to-fingertip distances
columns.extend([
    "dist_thumb",
    "dist_index",
    "dist_middle",
    "dist_ring",
    "dist_pinky"
])

# 10 fingertip-to-fingertip distances
columns.extend([
    "thumb_index",
    "thumb_middle",
    "thumb_ring",
    "thumb_pinky",
    "index_middle",
    "index_ring",
    "index_pinky",
    "middle_ring",
    "middle_pinky",
    "ring_pinky"
])

# 15 finger joint angles
for i in range(15):
    columns.append(f"angle_{i+1}")

columns.append("label")

df = pd.DataFrame(dataset, columns=columns)

df.to_csv(
    CSV_PATH,
    index=False
)

with open(LABEL_FILE, "w") as f:
    for label in sorted(labels):
        f.write(label + "\n")

hands.close()

print("\n" + "=" * 60)
print("Dataset Creation Completed")
print("=" * 60)

print(f"Total Samples      : {processed}")
print(f"Skipped Images     : {skipped}")
print(f"Number of Classes  : {len(labels)}")
print(f"CSV Saved          : {CSV_PATH}")
print(f"Labels Saved       : {LABEL_FILE}")
print("=" * 60)