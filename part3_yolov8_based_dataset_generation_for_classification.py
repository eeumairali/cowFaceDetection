import os
import cv2
import random
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
from ultralytics import YOLO
import streamlit as st
import datetime

class CowFaceDatasetPreparer:
    def __init__(self, video_folder, output_folder, model_path=None, frames_per_second=5, test_size=0.2, max_cows=10, trained_dir="models/yolov8_trained"):
        self.video_folder = video_folder
        self.output_folder = output_folder
        self.train_folder = os.path.join(output_folder, "train")
        self.test_folder = os.path.join(output_folder, "test")
        self.trained_dir = trained_dir
        if model_path is None:
            from part1_yolov8_enhanced_dataset import YOLOTrainer
            available_models = YOLOTrainer.list_trained_models(self.trained_dir)
            if not available_models:
                st.error(f"No trained YOLO models found in {self.trained_dir}.")
                raise FileNotFoundError(f"No trained YOLO models found in {self.trained_dir}.")
            model_path = st.selectbox("Select a trained YOLO model", available_models, key="yolo_model_select_dataset")
            model_path = os.path.join(self.trained_dir, model_path)
        self.model = YOLO(model_path)
        self.frames_per_second = frames_per_second
        self.test_size = test_size
        self.max_cows = max_cows
        
        os.makedirs(self.train_folder, exist_ok=True)
        os.makedirs(self.test_folder, exist_ok=True)
        
        self.cow_faces = {}
    
    def extract_faces(self):
        video_files = [f for f in os.listdir(self.video_folder) if f.endswith((".mp4", ".avi", ".mov"))]
        for video_file in video_files:
            video_path = os.path.join(self.video_folder, video_file)
            cow_id = video_file.split(".")[0]
            if cow_id not in self.cow_faces:
                self.cow_faces[cow_id] = []
            cap = cv2.VideoCapture(video_path)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            frame_interval = max(1, fps // self.frames_per_second)
            frame_count = 0
            with st.spinner(f"Extracting faces from {video_file}..."):
                while cap.isOpened():
                    success, frame = cap.read()
                    if not success:
                        break
                    if frame_count % frame_interval == 0:
                        results = self.model.predict(frame)
                        for result in results:
                            for box in result.boxes.xyxy:
                                x_min, y_min, x_max, y_max = map(int, box)
                                cropped_face = frame[y_min:y_max, x_min:x_max]
                                self.cow_faces[cow_id].append(cropped_face)
                    frame_count += 1
            cap.release()

    def split_dataset(self):
        selected_cows = random.sample(list(self.cow_faces.keys()), min(self.max_cows, len(self.cow_faces)))
        for cow_id in selected_cows:
            images = self.cow_faces[cow_id]
            train_imgs, test_imgs = train_test_split(images, test_size=self.test_size, random_state=42)
            cow_train_folder = os.path.join(self.train_folder, cow_id)
            cow_test_folder = os.path.join(self.test_folder, cow_id)
            os.makedirs(cow_train_folder, exist_ok=True)
            os.makedirs(cow_test_folder, exist_ok=True)
            # Save train images
            for idx, img in enumerate(train_imgs):
                img_path = os.path.join(cow_train_folder, f"{idx}.jpg")
                cv2.imwrite(img_path, img)
            # Save test images
            for idx, img in enumerate(test_imgs):
                img_path = os.path.join(cow_test_folder, f"{idx}.jpg")
                cv2.imwrite(img_path, img)

    def prepare_dataset(self):
        self.extract_faces()
        self.split_dataset()
        st.success(f"✅ Dataset preparation complete! Extracted and sorted images for {len(self.cow_faces)} cows.")