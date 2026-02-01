import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Rescaling
from tensorflow.keras.models import Sequential, Model
import numpy as np
import cv2
import os

def build_model(input_shape=(128, 128, 3)):
    model = Sequential([
        Rescaling(1./255, input_shape=input_shape),
        Conv2D(32, (3,3), activation='relu'),
        MaxPooling2D(),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(),
        Conv2D(128, (3,3), activation='relu', name="last_conv"),
        MaxPooling2D(),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    return model

model = build_model()
model.load_weights("deepfake_model.h5")
_ = model(tf.zeros((1, 128, 128, 3)))
model.summary()

def crop_face(image):
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return image
    x, y, w, h = faces[0]
    return image[y:y+h, x:x+w]

def generate_gradcam(model, img, last_conv_layer="last_conv"):
    conv_layer = model.get_layer(last_conv_layer)

    grad_model = Model(
        inputs=model.inputs,
        outputs=[conv_layer.output, model.outputs]
    )

    img_tensor = np.expand_dims(img, axis=0).astype("float32") / 255.0
    img_tensor = tf.convert_to_tensor(img_tensor)

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_tensor, training=False)
        preds = preds[0]  # unpack list
        loss = preds[:, 0]

    grads = tape.gradient(loss, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_out = conv_out[0]

    heatmap = conv_out @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.nn.relu(heatmap)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-10)

    return heatmap.numpy()

def overlay_heatmap(original, heatmap):
    heatmap = cv2.resize(heatmap, (original.shape[1], original.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

def process_all_frames(model, frames_dir="frames", output_dir="gradcam_output"):
    os.makedirs(output_dir, exist_ok=True)
    valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

    for root, dirs, files in os.walk(frames_dir):
        for filename in files:
            if not filename.lower().endswith(valid_ext):
                print("Skipping:", filename)
                continue

            frame_path = os.path.join(root, filename)
            print("Processing:", frame_path)

            img = cv2.imread(frame_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            face = crop_face(img_rgb)
            face_resized = cv2.resize(face, (128, 128))

            heatmap = generate_gradcam(model, face_resized)
            result = overlay_heatmap(face, heatmap)

            rel = os.path.relpath(root, frames_dir)
            save_dir = os.path.join(output_dir, rel)
            os.makedirs(save_dir, exist_ok=True)

            save_path = os.path.join(save_dir, f"gradcam_{filename}")
            cv2.imwrite(save_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
            print("Saved:", save_path)

process_all_frames(model)