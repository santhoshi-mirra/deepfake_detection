import streamlit as st
import tempfile
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

# ----------------------------
# Simulated backend functions
# ----------------------------
def predict_video(video_path):
    import random
    return random.choice(['Real', 'Fake']), random.uniform(0.6, 0.99)

def frame_confidence_analysis(video_path):
    import random
    return [random.uniform(0, 1) for _ in range(50)]

def generate_gradcam(video_path, frame_indices=[10,20,30]):
    heatmaps = {}
    for idx in frame_indices:
        heatmaps[idx] = Image.fromarray(np.random.randint(0,255,(224,224,3),dtype=np.uint8))
    return heatmaps

# ----------------------------
# Utility functions
# ----------------------------
def plot_confidence(confidences):
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(range(len(confidences)), confidences, marker='o', color='red')
    ax.set_xlabel("Frame Index")
    ax.set_ylabel("Fake Confidence")
    ax.set_title("Temporal Confidence Plot")
    st.pyplot(fig)

def download_confidences(confidences):
    buf = io.StringIO()
    buf.write("Frame,Fake_Confidence\n")
    for i, val in enumerate(confidences):
        buf.write(f"{i},{val:.4f}\n")
    st.download_button("📥 Download Frame Confidence CSV", buf.getvalue(), file_name="frame_confidence.csv", mime="text/csv")

def display_gradcam_slider(gradcam_images):
    st.subheader("Grad-CAM Viewer")
    frames = list(gradcam_images.keys())
    selected_frame = st.select_slider("Select frame for Grad-CAM", options=frames)
    st.image(gradcam_images[selected_frame], caption=f"Frame {selected_frame} Grad-CAM", use_column_width=True)

def failure_analysis_section():
    st.subheader("Failure Case Analysis")
    st.markdown("""
    - Low-quality or blurred videos may reduce confidence.  
    - Grad-CAM may appear diffuse in uncertain regions.  
    - Confidence drop indicates model uncertainty.
    """)

def model_info_section():
    st.sidebar.subheader("Model Info")
    st.sidebar.markdown("""
    **Model:** Pretrained CNN-based DeepFake detector  
    **Input:** Video frames (faces cropped)  
    **Output:** Frame-wise fake confidence, video-level prediction  
    **Explainability:** Grad-CAM heatmaps highlight regions influencing predictions
    """)

# ----------------------------
# Page Config & CSS
# ----------------------------
st.set_page_config(page_title="DeepFake Detection", layout="wide")
st.markdown("""
<style>
.reportview-container, .main {background-color:white; color:black;}
.stButton>button {background-color:black; color:white; font-weight:bold; height:3em; width:100%;}
</style>
""", unsafe_allow_html=True)

st.title("🎬 DeepFake Detection System")
st.markdown("""
Upload a video to classify it as Real or Fake, visualize frame-wise confidence, explore Grad-CAM explainability, and analyze failure cases.
""")

# Sidebar
model_info_section()

# ----------------------------
# Video Upload Section
# ----------------------------
uploaded_file = st.file_uploader("Upload a .mp4 video", type=["mp4"])

if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name
    st.video(video_path)

    # Confidence Threshold Input
    threshold = st.slider("Fake Confidence Threshold (%)", min_value=50, max_value=99, value=80, step=1)

    # ----------------------------
    # Model Processing Spinner
    # ----------------------------
    with st.spinner("⏳ Analyzing video..."):
        label, confidence = predict_video(video_path)
        confidences = frame_confidence_analysis(video_path)
        gradcam_images = generate_gradcam(video_path)

    st.success("✅ Analysis Complete!")

    # ----------------------------
    # Video-level Prediction
    # ----------------------------
    st.subheader("Video-level Prediction")
    st.markdown(f"**Prediction:** {label}  \n**Confidence:** {confidence*100:.2f}%")
    if confidence*100 > threshold:
        st.info(f"Confidence above threshold ({threshold}%) - suspicious video")
    else:
        st.success(f"Confidence below threshold ({threshold}%) - likely real")

    # ----------------------------
    # Frame-wise Confidence Plot
    # ----------------------------
    st.subheader("Frame-wise Confidence Analysis")
    plot_confidence(confidences)
    download_confidences(confidences)

    # ----------------------------
    # Grad-CAM Viewer
    # ----------------------------
    display_gradcam_slider(gradcam_images)

    # ----------------------------
    # Failure Case Analysis
    # ----------------------------
    failure_analysis_section()

    # ----------------------------
    # Extra Utilities
    # ----------------------------
    st.subheader("Additional Tools")
    if st.button("🔄 Re-run Analysis"):
        st.experimental_rerun()

    st.markdown("""
    - Use the slider above to change confidence threshold.  
    - Download frame confidence CSV for offline analysis.  
    - Inspect Grad-CAM for suspicious frames.
    """)
else:
    st.warning("Please upload a video to start the analysis.")



