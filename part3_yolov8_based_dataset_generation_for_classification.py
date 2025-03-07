import os
import cv2
import random
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
from ultralytics import YOLO

class CowFaceDatasetPreparer:
    def __init__(self, video_folder, output_folder, model_path, frames_per_second=5, test_size=0.2, max_cows=10):
        self.video_folder = video_folder
        self.output_folder = output_folder
        self.train_folder = os.path.join(output_folder, "train")
        self.test_folder = os.path.join(output_folder, "test")
        self.model = YOLO(model_path)
        self.frames_per_second = frames_per_second
        self.test_size = test_size
        self.max_cows = max_cows
        
        os.makedirs(self.train_folder, exist_ok=True)
        os.makedirs(self.test_folder, exist_ok=True)
        
        self.cow_faces = {}
    
    def extract_faces(self):
        video_files = [f for f in os.listdir(self.video_folder) if f.endswith(('.mp4', '.avi', '.mov'))]
        
        for video_file in video_files:
            video_path = os.path.join(self.video_folder, video_file)
            cow_id = video_file.split(".")[0]  # Assuming file format "cowID_*.mp4"

            if cow_id not in self.cow_faces:
                self.cow_faces[cow_id] = []

            cap = cv2.VideoCapture(video_path)
            fps = int(cap.get(cv2.CAP_PROP_FPS))  # Get video FPS
            frame_interval = max(1, fps // self.frames_per_second)  # Capture every Nth frame

            frame_count = 0
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break

                if frame_count % frame_interval == 0:  # Extract frame at the right interval
                    results = self.model.predict(frame)  # Use YOLO to detect cow faces

                    for result in results:
                        for box in result.boxes.xyxy:  # Bounding boxes (x_min, y_min, x_max, y_max)
                            x_min, y_min, x_max, y_max = map(int, box)
                            cropped_face = frame[y_min:y_max, x_min:x_max]
                            
                            cow_folder = os.path.join(self.output_folder, cow_id)
                            os.makedirs(cow_folder, exist_ok=True)
                            
                            face_filename = f"{len(self.cow_faces[cow_id])}.jpg"
                            face_path = os.path.join(cow_folder, face_filename)
                            cv2.imwrite(face_path, cropped_face)
                            self.cow_faces[cow_id].append(face_path)
                
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

            for img in train_imgs:
                shutil.move(img, os.path.join(cow_train_folder, os.path.basename(img)))
            for img in test_imgs:
                shutil.move(img, os.path.join(cow_test_folder, os.path.basename(img)))
    
    def prepare_dataset(self):
        self.extract_faces()
        self.split_dataset()
        print(f"✅ Dataset preparation complete! Extracted and sorted images for {len(self.cow_faces)} cows.")

if __name__ == "__main__":
    video_folder = "/mnt/c/Users/eeuma/Desktop/cow_facial_recognition_yolo_imagenet/cow_faces_videos"
    output_folder = "/mnt/c/Users/eeuma/Desktop/cow_facial_recognition_yolo_imagenet/dataset_for_classification"
    model_path = "/mnt/c/Users/eeuma/Desktop/cow_facial_recognition_yolo_imagenet/models/fine_tuned/yolov8_fine_tuned_cow_face.pt"
    
    dataset_preparer = CowFaceDatasetPreparer(video_folder, output_folder, model_path)
    dataset_preparer.prepare_dataset()