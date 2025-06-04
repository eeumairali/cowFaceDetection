"""
🟦 YOLO+EfficientNet/ResNet Video Pipeline

🔹 **Models:** YOLOv8 (detection) + EfficientNet/ResNet (classification)
🔹 **Purpose:** Detects and classifies all cow faces in each frame of a video.
🔹 **Key Features:**
    - YOLOv8: Real-time detection, sliding window (default: 640x640), multi-scale.
    - EfficientNet/ResNet: Classifies each detected face per frame.
    - Draws bounding boxes and predicted labels on each frame.
    - Saves and displays the annotated video in Streamlit.
    - Handles multiple faces per frame, robust to video input.

🔹 **Technical Details:**
    - YOLOv8: ~37+ layers, 3x3/1x1 convolutions, SiLU activation.
    - EfficientNet-B0: 18 layers, MBConv blocks, Swish activation.
    - ResNet: 34-152 layers, skip connections, ReLU activation.
    - Output: Annotated video with bounding boxes and class labels.

🔹 **Usage:**
    - Upload a video, select models, and run the pipeline.
    - See both detection and classification results visually on video output.

"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import os
from ultralytics import YOLO
import streamlit as st
import datetime
from part5_resnet_model_predict import CowFacialRecognition

class CowFaceRecognizer:
    def __init__(self, train_dir, model_path=None, yolo_model_path=None, input_video_path=None, output_video_path=None, trained_dir_efficientnet="models/efficientnet_trained", trained_dir_yolo="models/yolov8_trained"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        st.info(f"Using device: {self.device}")
        self.train_dir = train_dir
        self.trained_dir_efficientnet = trained_dir_efficientnet
        self.trained_dir_yolo = trained_dir_yolo
        self.class_names = sorted(os.listdir(train_dir))
        self.img_size = 224
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        if model_path is None:
            available_models = CowFacialRecognition.list_trained_models(self.trained_dir_efficientnet)
            if not available_models:
                st.error(f"No trained EfficientNet models found in {self.trained_dir_efficientnet}.")
                raise FileNotFoundError(f"No trained EfficientNet models found in {self.trained_dir_efficientnet}.")
            model_path = st.selectbox("Select a trained EfficientNet model", available_models, key="efficientnet_model_select_vid")
            model_path = os.path.join(self.trained_dir_efficientnet, model_path)
        if yolo_model_path is None:
            from part1_yolov8_enhanced_dataset import YOLOTrainer
            available_yolo = YOLOTrainer.list_trained_models(self.trained_dir_yolo)
            if not available_yolo:
                st.error(f"No trained YOLO models found in {self.trained_dir_yolo}.")
                raise FileNotFoundError(f"No trained YOLO models found in {self.trained_dir_yolo}.")
            yolo_model_path = st.selectbox("Select a trained YOLO model", available_yolo, key="yolo_model_select_vid")
            yolo_model_path = os.path.join(self.trained_dir_yolo, yolo_model_path)
        self.classifier_model = self.load_classifier_model(model_path)
        self.yolo_model = YOLO(yolo_model_path)
        self.input_video_path = input_video_path
        self.output_video_path = output_video_path

    def load_classifier_model(self, model_path):
        num_classes = len(self.class_names)
        model = models.efficientnet_b0(weights=None)
        model.classifier = nn.Sequential(
            nn.Linear(model.classifier[1].in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        return model.to(self.device).eval()

    def load_font(self, font_size):
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "C:/Windows/Fonts/arial.ttf"
        ]
        for font_path in font_paths:
            if os.path.exists(font_path):
                return ImageFont.truetype(font_path, font_size)
        return ImageFont.load_default()

    def process_video(self):
        cap = cv2.VideoCapture(self.input_video_path)
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_video_path, fourcc, fps, (frame_width, frame_height))
        with st.spinner("Running YOLO+ResNet on video..."):
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(pil_image)
                results = self.yolo_model.predict(frame)
                for result in results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        box_height = y2 - y1
                        font_size = max(30, int(box_height * 0.05))
                        font = self.load_font(font_size)
                        face = pil_image.crop((x1, y1, x2, y2))
                        face = self.transform(face).unsqueeze(0).to(self.device)
                        with torch.no_grad():
                            outputs = self.classifier_model(face)
                            _, predicted = torch.max(outputs, 1)
                            predicted_class = self.class_names[predicted.item()]
                        draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=4)
                        label_x, label_y = x1, max(0, y1 - font_size - 10)
                        draw.text((label_x, label_y), predicted_class, fill="red", font=font)
                frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                out.write(frame)
            cap.release()
            out.release()
            cv2.destroyAllWindows()
        if self.output_video_path and os.path.exists(self.output_video_path):
            st.success(f"✅ Saved output video: {self.output_video_path}")
            st.video(self.output_video_path)