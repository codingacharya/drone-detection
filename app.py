import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import os
from PIL import Image
import glob

st.set_page_config(page_title="Drone Detection App", layout="wide")
st.title("🚁 Drone Detection App (YOLOv8)")

# ------------------------------
# Automatically find latest trained weights
weights_dir = "runs/detect/drone_detector/weights"
if not os.path.exists(weights_dir):
    st.error(f"❌ Weights folder not found at: {weights_dir}\nRun train_model.py first!")
    st.stop()

weight_files = glob.glob(os.path.join(weights_dir, "*.pt"))
if not weight_files:
    st.error(f"❌ No .pt weights found in {weights_dir}. Please train the model first!")
    st.stop()

# Choose the latest file by modification time
MODEL_PATH = max(weight_files, key=os.path.getmtime)
st.info(f"Using model weights: {MODEL_PATH}")

# Load YOLOv8 model
model = YOLO(MODEL_PATH)
st.success("✅ Model loaded successfully!")

# ------------------------------
# Confidence slider
conf_thres = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.25, 0.05)

# ------------------------------
# Draw bounding boxes
def draw_boxes(frame, results):
    count = 0
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        if conf < conf_thres:
            continue
        cls = int(box.cls[0])
        label = model.names[cls]
        count += 1
        # Draw rectangle and label
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 3)
        cv2.putText(frame, f"{label} {conf:.2f}", (int(x1), int(y1)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
    return frame, count

# ------------------------------
# Detection functions
def detect_image(img):
    results = model.predict(img, conf=conf_thres)[0]
    frame, count = draw_boxes(img.copy(), results)
    return frame, count

def detect_video(video_file):
    temp_path = "temp_video.mp4"
    with open(temp_path, "wb") as f:
        f.write(video_file.read())

    cap = cv2.VideoCapture(temp_path)
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(frame, conf=conf_thres)[0]
        frame, count = draw_boxes(frame, results)
        stframe.image(frame, channels="BGR")
    cap.release()

# ------------------------------
# Streamlit UI
mode = st.sidebar.selectbox("Select Mode", ["Image", "Video", "Webcam"])

if mode == "Image":
    file = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])
    if file:
        img = Image.open(file)
        img_np = np.array(img)
        st.image(img_np, caption="Uploaded Image", use_column_width=True)
        if st.button("Detect Drone"):
            result, count = detect_image(img_np)
            st.image(result, caption=f"Detected ({count} drones)", use_column_width=True)

elif mode == "Video":
    file = st.file_uploader("Upload a video", type=["mp4","avi"])
    if file:
        st.video(file)
        if st.button("Start Detection"):
            detect_video(file)

elif mode == "Webcam":
    if st.button("Start Webcam"):
        cap = cv2.VideoCapture(0)
        stframe = st.empty()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = model.predict(frame, conf=conf_thres)[0]
            frame, count = draw_boxes(frame, results)
            stframe.image(frame, channels="BGR")
        cap.release()
