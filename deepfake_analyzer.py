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
         Frame-wise confidence storage
         Video-level prediction
        
        Extracts frames, predicts each frame, stores confidences,
        and aggregates for final video-level prediction.
        """
        cap = cv2.VideoCapture(video_path)
        
        self.frame_confidences = []
        self.frame_indices = []
        count = 0
        frame_idx = 0
        
        print(f"\n{'='*60}")
        print(f"Processing: {os.path.basename(video_path)}")
        print(f"{'='*60}")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or len(self.frame_confidences) >= MAX_FRAMES:
                break
            
            if count % FRAME_GAP == 0:
                # Preprocess frame
                processed_frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
                processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                processed_frame = processed_frame / 255.0
                processed_frame = np.expand_dims(processed_frame, axis=0)
                
                #  Predict and store fake confidence for this frame
                fake_confidence = self.model.predict(processed_frame, verbose=0)[0][0]
                self.frame_confidences.append(float(fake_confidence))
                self.frame_indices.append(frame_idx)
                
                frame_idx += 1
            
            count += 1
        
        cap.release()
        
        # Aggregate predictions for video-level decision
        return self.aggregate_predictions()
    
    def aggregate_predictions(self):
        """
        Video-level prediction
        Aggregate frame-wise confidences using multiple metrics
        """
        if not self.frame_confidences:
            print("⚠ No frames processed!")
            return None
        
        confidences = np.array(self.frame_confidences)
        
        # Calculate aggregation metrics
        mean_confidence = np.mean(confidences)
        max_confidence = np.max(confidences)
        median_confidence = np.median(confidences)
        min_confidence = np.min(confidences)
        std_confidence = np.std(confidences)
        
        # Improved video-level decision using temporal behavior
        fake_frames_ratio = np.sum(confidences > 0.6) / len(confidences)

        if mean_confidence > 0.6 and fake_frames_ratio > 0.4 and std_confidence > 0.15:
          final_prediction = "FAKE"
          confidence_percentage = mean_confidence * 100
        else:
          final_prediction = "REAL"
          confidence_percentage = (1 - mean_confidence) * 100
        
        results = {
            "final_prediction": final_prediction,
            "confidence_percentage": confidence_percentage,
            "mean_confidence": mean_confidence,
            "max_confidence": max_confidence,
            "min_confidence": min_confidence,
            "median_confidence": median_confidence,
            "std_confidence": std_confidence,
            "num_frames": len(self.frame_confidences)
        }
        
        return results
    
    def plot_temporal_analysis(self, video_name="video", save_path=None):
        """
        Temporal confidence analysis
        Plot confidence vs frame index to observe instability patterns.
        Fake videos often show more temporal instability.
        """
        if not self.frame_confidences:
            print("⚠ No frame confidences to plot!")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Plot 1: Confidence over time
        ax1.plot(self.frame_indices, self.frame_confidences, 
                marker='o', linestyle='-', linewidth=2, markersize=5, 
                color='#2E86AB', label='Frame Confidence')
        
        # Add threshold line
        ax1.axhline(y=0.5, color='red', linestyle='--', 
                   label='Decision Threshold', linewidth=2.5)
        
        # Add mean line
        mean_conf = np.mean(self.frame_confidences)
        ax1.axhline(y=mean_conf, color='green', linestyle='--', 
                   label=f'Mean Confidence ({mean_conf:.3f})', linewidth=2)
        
        # Shade regions
        ax1.fill_between(self.frame_indices, 0, 0.5, alpha=0.1, color='blue', label='REAL Region')
        ax1.fill_between(self.frame_indices, 0.5, 1, alpha=0.1, color='red', label='FAKE Region')
        
        ax1.set_xlabel('Frame Index', fontsize=13, fontweight='bold')
        ax1.set_ylabel('Fake Confidence', fontsize=13, fontweight='bold')
        ax1.set_title(f'Temporal Confidence Analysis: {video_name}', 
                     fontsize=15, fontweight='bold', pad=20)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.legend(fontsize=10, loc='upper right')
        ax1.set_ylim(0, 1)
        
        # Add statistics box
        std_conf = np.std(self.frame_confidences)
        stats_text = f'Statistics:\n'
        stats_text += f'Mean: {mean_conf:.4f}\n'
        stats_text += f'Std Dev: {std_conf:.4f}\n'
        stats_text += f'Min: {np.min(self.frame_confidences):.4f}\n'
        stats_text += f'Max: {np.max(self.frame_confidences):.4f}\n'
        stats_text += f'\nInstability: {"HIGH" if std_conf > 0.15 else "LOW"}'
        
        ax1.text(0.02, 0.98, stats_text, 
                transform=ax1.transAxes, 
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=10, family='monospace')
        
        # Plot 2: Histogram of confidences
        ax2.hist(self.frame_confidences, bins=20, color='#A23B72', 
                alpha=0.7, edgecolor='black', linewidth=1.2)
        ax2.axvline(x=0.5, color='red', linestyle='--', 
                   label='Threshold', linewidth=2.5)
        ax2.axvline(x=mean_conf, color='green', linestyle='--', 
                   label=f'Mean ({mean_conf:.3f})', linewidth=2)
        ax2.set_xlabel('Fake Confidence', fontsize=13, fontweight='bold')
        ax2.set_ylabel('Frequency', fontsize=13, fontweight='bold')
        ax2.set_title('Distribution of Frame Confidences', 
                     fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Plot saved to: {save_path}")
        
        plt.show()
    
    def print_results(self, results):
        """Print detailed analysis results"""
        print("\n" + "="*60)
        print("VIDEO-LEVEL PREDICTION RESULTS")
        print("="*60)
        print(f"Frames Analyzed: {results['num_frames']}")
        print(f"\nAggregation Metrics:")
        print(f"  Mean Confidence:   {results['mean_confidence']:.4f}")
        print(f"  Median Confidence: {results['median_confidence']:.4f}")
        print(f"  Max Confidence:    {results['max_confidence']:.4f}")
        print(f"  Min Confidence:    {results['min_confidence']:.4f}")
        print(f"  Std Deviation:     {results['std_confidence']:.4f}")
        
        # Instability interpretation
        if results['std_confidence'] > 0.15:
            instability = "HIGH (Likely FAKE - inconsistent predictions)"
        else:
            instability = "LOW (Likely REAL - consistent predictions)"
        print(f"  Instability:       {instability}")
        
        print(f"\n{'='*60}")
        print(f"FINAL PREDICTION: {results['final_prediction']}")
        print(f"Confidence: {results['confidence_percentage']:.2f}%")
        print("="*60 + "\n")


# Main execution
if __name__ == "__main__":
    analyzer = DeepfakeAnalyzer(MODEL_PATH)

    # Dataset folder (FAKE or REAL)
    DATASET_DIR = "dataset"

    # Get all video files
    test_videos = []

for label in ["real", "fake"]:
    class_dir = os.path.join(DATASET_DIR, label)
    for f in os.listdir(class_dir):
        if f.lower().endswith((".mp4", ".avi", ".mov")):
            test_videos.append(os.path.join(class_dir, f))

    if not test_videos:
        print(" No video files found in the directory!")

    for video_path in test_videos:
        print(f"\n Analyzing video: {video_path}")

        results = analyzer.predict_video(video_path)

        if results:
            analyzer.print_results(results)

            video_name = os.path.basename(video_path)
            save_path = f"temporal_analysis_{video_name.split('.')[0]}.png"
            analyzer.plot_temporal_analysis(
                video_name=video_name,
                save_path=save_path
            )
