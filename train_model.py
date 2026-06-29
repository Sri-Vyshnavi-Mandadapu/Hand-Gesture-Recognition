"""
train_model.py

TensorFlow Hand Gesture Training
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    BatchNormalization
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint
)

from config import (
    CSV_PATH,
    MODEL_PATH,
    SCALER_PATH,
    ENCODER_PATH,
    METRICS_PATH,
    TEST_SIZE,
    RANDOM_STATE,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    MODELS_PATH
)

os.makedirs(MODELS_PATH, exist_ok=True)

print("="*60)
print("Loading Dataset")
print("="*60)

df = pd.read_csv(CSV_PATH)

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

print("Samples :", len(X))
print("Features:", X.shape[1])

# ----------------------------------------------------
# Label Encoding
# ----------------------------------------------------

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(y)

joblib.dump(
    label_encoder,
    ENCODER_PATH
)

# ----------------------------------------------------
# Feature Scaling
# ----------------------------------------------------

scaler = StandardScaler()

X = scaler.fit_transform(X)

joblib.dump(
    scaler,
    SCALER_PATH
)

# ----------------------------------------------------
# Train Test Split
# ----------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)
# ----------------------------------------------------
# Build Neural Network
# ----------------------------------------------------
# ----------------------------------------------------
# Build Neural Network
# ----------------------------------------------------

num_classes = len(label_encoder.classes_)

model = Sequential([

    Dense(512, activation="relu", input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.40),

    Dense(256, activation="relu"),
    BatchNormalization(),
    Dropout(0.30),

    Dense(128, activation="relu"),
    BatchNormalization(),
    Dropout(0.30),

    Dense(64, activation="relu"),
    BatchNormalization(),
    Dropout(0.20),

    Dense(num_classes, activation="softmax")

])
optimizer = tf.keras.optimizers.Adam(
    learning_rate=LEARNING_RATE
)

model.compile(
    optimizer=optimizer,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ----------------------------------------------------
# Callbacks
# ----------------------------------------------------

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True,
    verbose=1
)

checkpoint = ModelCheckpoint(
    MODEL_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

# ----------------------------------------------------
# Train
# ----------------------------------------------------

print("\n" + "=" * 60)
print("Training Started...")
print("=" * 60)

history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop, checkpoint],
    verbose=1
)

print("\nTraining Completed.")
# ----------------------------------------------------
# Evaluation
# ----------------------------------------------------

print("\n" + "=" * 60)
print("Evaluating Model")
print("=" * 60)

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print(f"Test Accuracy : {test_accuracy:.4f}")
print(f"Test Loss     : {test_loss:.4f}")

# ----------------------------------------------------
# Predictions
# ----------------------------------------------------

y_pred_prob = model.predict(X_test, verbose=0)

y_pred = np.argmax(y_pred_prob, axis=1)

accuracy = accuracy_score(y_test, y_pred)

print(f"\nFinal Accuracy : {accuracy*100:.2f}%")

# ----------------------------------------------------
# Classification Report
# ----------------------------------------------------

report = classification_report(
    y_test,
    y_pred,
    target_names=label_encoder.classes_,
    output_dict=True
)

print("\nClassification Report\n")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_
    )
)

# ----------------------------------------------------
# Confusion Matrix
# ----------------------------------------------------

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8,8))
plt.imshow(cm, interpolation="nearest")
plt.title("Confusion Matrix")
plt.colorbar()

ticks = np.arange(len(label_encoder.classes_))

plt.xticks(
    ticks,
    label_encoder.classes_,
    rotation=45
)

plt.yticks(
    ticks,
    label_encoder.classes_
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

plt.savefig(
    os.path.join(
        MODELS_PATH,
        "confusion_matrix.png"
    )
)

plt.close()

# ----------------------------------------------------
# Accuracy Graph
# ----------------------------------------------------

plt.figure(figsize=(8,5))

plt.plot(
    history.history["accuracy"],
    label="Training"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation"
)

plt.title("Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        MODELS_PATH,
        "accuracy.png"
    )
)

plt.close()

# ----------------------------------------------------
# Loss Graph
# ----------------------------------------------------

plt.figure(figsize=(8,5))

plt.plot(
    history.history["loss"],
    label="Training"
)

plt.plot(
    history.history["val_loss"],
    label="Validation"
)

plt.title("Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        MODELS_PATH,
        "loss.png"
    )
)

plt.close()

# ----------------------------------------------------
# Metrics JSON
# ----------------------------------------------------

metrics = {
    "accuracy": float(accuracy),
    "loss": float(test_loss),
    "classes": list(label_encoder.classes_),
    "samples": int(len(X)),
    "features": int(X.shape[1])
}

with open(METRICS_PATH, "w") as f:
    json.dump(
        metrics,
        f,
        indent=4
    )

print("\nMetrics saved.")

# ----------------------------------------------------
# Save Final Model
# ----------------------------------------------------

model.save(MODEL_PATH)

print("\nModel saved successfully.")

print("\n" + "=" * 60)
print("Training Completed Successfully")
print("=" * 60)