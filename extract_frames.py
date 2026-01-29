import cv2
import os

INPUT_DIR = "dataset"
OUTPUT_DIR = "frames"
FRAME_GAP = 5
MAX_FRAMES = 150

os.makedirs(OUTPUT_DIR, exist_ok=True)

for label in ["real", "fake"]:
    input_path = os.path.join(INPUT_DIR, label)
    output_path = os.path.join(OUTPUT_DIR, label)
    os.makedirs(output_path, exist_ok=True)

    for video_name in os.listdir(input_path):
        video_path = os.path.join(input_path, video_name)
        cap = cv2.VideoCapture(video_path)

        count = 0
        saved = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or saved >= MAX_FRAMES:
                break

            if count % FRAME_GAP == 0:
                frame = cv2.resize(frame, (128, 128))
                frame_name = f"{video_name}_{saved}.jpg"
                cv2.imwrite(os.path.join(output_path, frame_name), frame)
                saved += 1

            count += 1

        cap.release()
