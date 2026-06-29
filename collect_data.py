"""
collect_data.py
----------------
Professional Dataset Collection Tool

Features:
- Left & Right hand detection
- Cropped hand image saving
- Manual & Auto capture
- FPS counter
- Progress display

Author: Sri Vyshnavi Mandadapu
"""

import os
import time
import cv2
import mediapipe as mp

from config import (
    DATASET_PATH,
    CAMERA_ID,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    MAX_HANDS,
    DETECTION_CONFIDENCE,
    TRACKING_CONFIDENCE
)

from utils import (
    create_directories,
    get_bounding_box,
    crop_hand,
    FPS
)

# =====================================================
# Initial Setup
# =====================================================

create_directories()

gesture_name = input("\nEnter Gesture Name : ").strip()

if gesture_name == "":
    print("Gesture name cannot be empty.")
    exit()

gesture_folder = os.path.join(DATASET_PATH, gesture_name)
os.makedirs(gesture_folder, exist_ok=True)

image_count = len([
    f for f in os.listdir(gesture_folder)
    if f.endswith(".jpg")
])

print(f"\nImages already present : {image_count}")

# =====================================================
# MediaPipe Initialization
# =====================================================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=MAX_HANDS,
    min_detection_confidence=DETECTION_CONFIDENCE,
    min_tracking_confidence=TRACKING_CONFIDENCE
)

# =====================================================
# Camera Initialization
# =====================================================

cap = cv2.VideoCapture(CAMERA_ID)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

if not cap.isOpened():
    print("Unable to access camera.")
    exit()

# =====================================================
# Variables
# =====================================================

fps = FPS()

auto_capture = False

capture_interval = 0.20

last_capture = time.time()

TARGET_SIZE = 224

FONT = cv2.FONT_HERSHEY_SIMPLEX

print("\n===================================")
print(" Dataset Collection Started")
print("===================================")
print("S : Save one image")
print("A : Toggle Auto Capture")
print("Q : Quit")
print("===================================\n")
# =====================================================
# Main Loop
# =====================================================

while True:

    success, frame = cap.read()

    if not success:
        print("Failed to capture frame.")
        break

    # Mirror effect
    frame = cv2.flip(frame, 1)

    display = frame.copy()

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    current_time = time.time()

    detected_hand = False

    cropped_hand = None

    # -----------------------------------------------
    # Hand Detection
    # -----------------------------------------------

    if results.multi_hand_landmarks:

        for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks,
                results.multi_handedness):

            detected_hand = True

            hand_label = handedness.classification[0].label

            # Draw landmarks
            mp_draw.draw_landmarks(
                display,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Bounding Box
            bbox = get_bounding_box(
                hand_landmarks,
                display.shape,
                padding=25
            )

            x1, y1, x2, y2 = bbox

            # Draw rectangle
            cv2.rectangle(
                display,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Crop Hand
            cropped_hand = crop_hand(frame, bbox)

            if cropped_hand.size != 0:

                cropped_hand = cv2.resize(
                    cropped_hand,
                    (TARGET_SIZE, TARGET_SIZE)
                )

            # Display Left / Right
            cv2.putText(
                display,
                f"{hand_label} Hand",
                (x1, y1 - 10),
                FONT,
                0.7,
                (255, 0, 255),
                2
            )

            # We process one hand image at a time for dataset collection
            break

    # -----------------------------------------------
    # Information Panel
    # -----------------------------------------------

    cv2.rectangle(display, (10, 10), (360, 170), (0, 0, 0), -1)

    cv2.putText(
        display,
        f"Gesture : {gesture_name}",
        (20, 40),
        FONT,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        display,
        f"Images : {image_count}",
        (20, 70),
        FONT,
        0.7,
        (255, 255, 0),
        2
    )

    mode = "AUTO" if auto_capture else "MANUAL"

    cv2.putText(
        display,
        f"Mode : {mode}",
        (20, 100),
        FONT,
        0.7,
        (0, 255, 255),
        2
    )

    cv2.putText(
        display,
        f"FPS : {fps.update()}",
        (20, 130),
        FONT,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        display,
        "S=Save  A=Auto  Q=Quit",
        (20, 160),
        FONT,
        0.6,
        (180, 180, 180),
        1
    )

    # -----------------------------------------------
    # Show Camera
    # -----------------------------------------------

    cv2.imshow(
        "AI Hand Gesture Dataset Collector",
        display
    )
    # -----------------------------------------------
    # Keyboard Controls
    # -----------------------------------------------

    key = cv2.waitKey(1) & 0xFF

    # Manual Save
    if key == ord("s"):

        if cropped_hand is not None:

            filename = os.path.join(
                gesture_folder,
                f"{image_count:05d}.jpg"
            )

            cv2.imwrite(
                filename,
                cropped_hand
            )

            image_count += 1

            print(f"[Saved] {filename}")

        else:
            print("No hand detected.")

    # Toggle Auto Capture
    elif key == ord("a"):

        auto_capture = not auto_capture

        if auto_capture:
            print("Auto Capture Enabled")
        else:
            print("Auto Capture Disabled")

    # Quit
    elif key == ord("q"):

        print("\nStopping Dataset Collection...")
        break

    # -----------------------------------------------
    # Auto Capture
    # -----------------------------------------------

    if auto_capture and detected_hand and cropped_hand is not None:

        if current_time - last_capture >= capture_interval:

            filename = os.path.join(
                gesture_folder,
                f"{image_count:05d}.jpg"
            )

            cv2.imwrite(
                filename,
                cropped_hand
            )

            image_count += 1

            last_capture = current_time

# =====================================================
# Cleanup
# =====================================================

cap.release()

hands.close()

cv2.destroyAllWindows()

print("\n===================================")
print("Dataset Collection Completed")
print(f"Total Images : {image_count}")
print("===================================")