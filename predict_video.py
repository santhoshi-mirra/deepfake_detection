import cv2
import numpy as np
import tensorflow as tf

MODEL_PATH = "deepfake_model.h5"
VIDEO_PATH = "dataset/fake/02_15__talking_against_wall__P2FAY3DR.mp4"
FRAME_GAP = 5
MAX_FRAMES = 150
IMG_SIZE = 128

model = tf.keras.models.load_model(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)

preds = []
count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret or len(preds) >= MAX_FRAMES:
        break

    if count % FRAME_GAP == 0:
        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frame = frame / 255.0
        frame = np.expand_dims(frame, axis=0)

        pred = model.predict(frame, verbose=0)[0][0]
        preds.append(pred)

    count += 1

cap.release()

avg_conf = np.mean(preds)

print("Average confidence:", avg_conf)
print("Prediction:", "FAKE" if avg_conf > 0.5 else "REAL")
