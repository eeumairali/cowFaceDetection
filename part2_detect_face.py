"""
🟦 YOLOv8 Face Detection Script

🔹 **Model:** YOLOv8 (You Only Look Once, v8)
🔹 **Purpose:** Detects and localizes cow faces in images and videos.
🔹 **Key Features:**
    - Real-time object detection with sliding window (default: 640x640).
    - Uses convolutional layers (kernels: 3x3, 1x1) for feature extraction.
    - Outputs bounding boxes, class labels, and confidence scores.
    - Fast inference, suitable for video streams.
    - Model variants: YOLOv8n, YOLOv8s, etc. (trade-off between speed and accuracy).

🔹 **Technical Details:**
    - Input window size: 640x640 (default, configurable).
    - Convolutional kernel sizes: 3x3, 1x1.
    - Number of layers: ~37 (YOLOv8n), more for larger variants.
    - Activation: SiLU (Swish).
    - Output: [x, y, w, h, confidence, class probabilities].

🔹 **Usage:**
    - Loads a trained YOLOv8 model and applies it to images/videos.
    - Saves and displays detected faces with bounding boxes.

"""

import cv2
import os
from ultralytics import YOLO
import streamlit as st
import datetime

class YOLOProcessor:
    def __init__(self, model_path=None, trained_dir="models/yolov8_trained"):
        """
        Initialize the YOLO model.
        :param model_path: Path to the trained YOLO model.
        """
        self.trained_dir = trained_dir
        if model_path is None:
            from part1_yolov8_enhanced_dataset import YOLOTrainer
            available_models = YOLOTrainer.list_trained_models(self.trained_dir)
            if not available_models:
                st.error(f"No trained YOLO models found in {self.trained_dir}.")
                raise FileNotFoundError(f"No trained YOLO models found in {self.trained_dir}.")
            model_path = st.selectbox("Select a trained YOLO model", available_models, key="yolo_model_select")
            model_path = os.path.join(self.trained_dir, model_path)
        self.model = YOLO(model_path)

    def process_image(self, image_path, output_folder="detected_face_yolov8"):
        """
        Process an image with YOLO and save the result.
        """
        os.makedirs(output_folder, exist_ok=True)
        with st.spinner("Detecting face in image..."):
            results = self.model(image_path)  # Run YOLO detection
            for r in results:
                r.save(filename=os.path.join(output_folder, "part2_demo_out.png"))
        st.success(f"Processed image saved at {output_folder}/part2_demo_out.png")

    def process_video(self, video_path, output_folder="detected_face_yolov8"):
        """
        Process a video frame-by-frame with YOLO and save the result.
        """
        os.makedirs(output_folder, exist_ok=True)
        
        # Load video
        cap = cv2.VideoCapture(video_path)
        frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        output_video_path = os.path.join(output_folder, "part2_demo_out.mp4")
        out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))

        with st.spinner("Detecting faces in video..."):
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Run YOLO detection
                results = self.model(frame)
                for r in results:
                    processed_frame = r.plot()  # Get processed frame with bounding boxes

                out.write(processed_frame)  # Save to output video

            cap.release()
            out.release()
        st.success(f"Processed video saved at {output_video_path}")
