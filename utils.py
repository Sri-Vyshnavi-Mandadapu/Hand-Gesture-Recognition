"""
utils.py

Utility functions for AI Hand Gesture Recognition

Author: Sri Vyshnavi Mandadapu
"""

import os
import csv
import cv2
import time
import numpy as np
from collections import deque, Counter
def create_directories():
    """
    Creates required project directories if they do not exist.
    """

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
def normalize_landmarks(hand_landmarks):
    """
    Normalize landmarks using wrist as origin.
    """

    wrist = hand_landmarks.landmark[0]

    features = []

    for lm in hand_landmarks.landmark:

        features.extend([
            lm.x - wrist.x,
            lm.y - wrist.y,
            lm.z - wrist.z
        ])

    return np.array(features)
def load_labels(label_file):

    with open(label_file, "r") as file:

        labels = [
            line.strip()
            for line in file.readlines()
        ]

    return labels

class PredictionHistory:

    def __init__(self, size=10):

        self.history = deque(maxlen=size)

    def update(self, prediction):

        self.history.append(prediction)

        return Counter(self.history).most_common(1)[0][0]
class FPS:

    def __init__(self):

        self.previous = time.time()

    def get(self):

        current = time.time()

        fps = 1 / (current - self.previous)

        self.previous = current

        return int(fps)



def log_prediction(
        logfile,
        hand,
        gesture,
        confidence
):

    file_exists = os.path.isfile(logfile)

    with open(logfile, "a", newline="") as file:

        writer = csv.writer(file)

        if not file_exists:

            writer.writerow([
                "Hand",
                "Gesture",
                "Confidence"
            ])

        writer.writerow([
            hand,
            gesture,
            confidence
        ])

def save_screenshot(frame, folder):

    filename = os.path.join(
        folder,
        f"{int(time.time())}.png"
    )

    cv2.imwrite(filename, frame)

    return filename

import mediapipe as mp

mp_draw = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


def draw_hand(image, hand_landmarks):

    mp_draw.draw_landmarks(
        image,
        hand_landmarks,
        mp_hands.HAND_CONNECTIONS
    )
def get_hand_label(handedness):

    return handedness.classification[0].label
