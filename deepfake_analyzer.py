import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import models
import matplotlib.pyplot as plt
import os

MODEL_PATH = "deepfake_model.h5"
FRAME_GAP = 5
MAX_FRAMES = 150
IMG_SIZE = 128


class DeepfakeAnalyzer:
    def __init__(self, model_path):
        """Load the trained model"""
        self.model = models.load_model(model_path)
        self.frame_confidences = []
        self.frame_indices = []

    def predict_video(self, video_path):
        """
        Extract frames, predict each frame, store confidences,
        and aggregate for final video-level prediction.
        """
        cap = cv2.VideoCapture(video_path)

        self.frame_confidences = []
        self.frame_indices = []
        count = 0
        frame_idx = 0

        print("\n" + "=" * 60)
        print(f"Processing: {os.path.basename(video_path)}")
        print("=" * 60)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or len(self.frame_confidences) >= MAX_FRAMES:
                break

            if count % FRAME_GAP == 0:
                frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = frame / 255.0
                frame = np.expand_dims(frame, axis=0)

                fake_confidence = self.model.predict(frame, verbose=0)[0][0]
                self.frame_confidences.append(float(fake_confidence))
                self.frame_indices.append(frame_idx)
                frame_idx += 1

            count += 1

        cap.release()
        return self.aggregate_predictions()

    def aggregate_predictions(self):
        """Video-level prediction using temporal behavior"""
        confidences = np.array(self.frame_confidences)

        if len(confidences) == 0:
            return None

        mean_conf = np.mean(confidences)
        std_conf = np.std(confidences)
        fake_ratio = np.sum(confidences > 0.6) / len(confidences)

        # FINAL LOGIC (TEMPORAL-BASED)
        if std_conf > 0.15 and fake_ratio > 0.4:
            final_prediction = "FAKE"
            confidence_percentage = mean_conf * 100
        else:
            final_prediction = "REAL"
            confidence_percentage = (1 - mean_conf) * 100

        return {
            "final_prediction": final_prediction,
            "confidence_percentage": confidence_percentage,
            "mean_confidence": mean_conf,
            "std_confidence": std_conf,
            "min_confidence": np.min(confidences),
            "max_confidence": np.max(confidences),
            "num_frames": len(confidences)
        }

    def plot_temporal_analysis(self, video_name, save_path=None):
        """Plot temporal confidence analysis"""
        confidences = self.frame_confidences
        indices = self.frame_indices

        plt.figure(figsize=(14, 6))
        plt.plot(indices, confidences, marker="o", linewidth=2)
        plt.axhline(0.5, color="red", linestyle="--", label="Threshold")
        plt.axhline(np.mean(confidences), color="green", linestyle="--", label="Mean")
        plt.ylim(0, 1)
        plt.xlabel("Frame Index")
        plt.ylabel("Fake Confidence")
        plt.title(f"Temporal Confidence Analysis: {video_name}")
        plt.legend()
        plt.grid(True)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved plot → {save_path}")

        plt.show()

    def print_results(self, results):
        print("\n" + "=" * 60)
        print("VIDEO-LEVEL RESULTS")
        print("=" * 60)
        print(f"Frames analyzed: {results['num_frames']}")
        print(f"Mean confidence: {results['mean_confidence']:.4f}")
        print(f"Std deviation:   {results['std_confidence']:.4f}")
        print(f"Min confidence:  {results['min_confidence']:.4f}")
        print(f"Max confidence:  {results['max_confidence']:.4f}")
        print("\nFINAL PREDICTION:", results["final_prediction"])
        print(f"Confidence: {results['confidence_percentage']:.2f}%")
        print("=" * 60 + "\n")


# ================== MAIN ==================

if __name__ == "__main__":
    analyzer = DeepfakeAnalyzer(MODEL_PATH)

    DATASET_DIR = "dataset"
    test_videos = []

    for label in ["real", "fake"]:
        class_dir = os.path.join(DATASET_DIR, label)
        for file in os.listdir(class_dir):
            if file.lower().endswith((".mp4", ".avi", ".mov")):
                test_videos.append(os.path.join(class_dir, file))

    if not test_videos:
        print("No video files found.")
        exit()

    for video in test_videos:
        results = analyzer.predict_video(video)
        if results:
            analyzer.print_results(results)
            analyzer.plot_temporal_analysis(
                video_name=os.path.basename(video),
                save_path=f"temporal_{os.path.basename(video)}.png"
            )
