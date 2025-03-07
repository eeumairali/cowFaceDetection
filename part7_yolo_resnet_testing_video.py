import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import os
from ultralytics import YOLO

class CowFaceRecognizer:
    def __init__(self, train_dir, model_path, yolo_model_path, input_video_path, output_video_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

        self.train_dir = train_dir
        self.model_path = model_path
        self.yolo_model_path = yolo_model_path
        self.input_video_path = input_video_path
        self.output_video_path = output_video_path

        self.class_names = sorted(os.listdir(train_dir))  # Ensure consistent indexing
        self.img_size = 224
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.classifier_model = self.load_classifier_model()
        self.yolo_model = YOLO(self.yolo_model_path)

    def load_classifier_model(self):
        num_classes = len(self.class_names)
        model = models.efficientnet_b0(weights=None)
        model.classifier = nn.Sequential(
            nn.Linear(model.classifier[1].in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        return model.to(self.device).eval()

    def load_font(self, font_size):
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Linux
            "C:/Windows/Fonts/arial.ttf"  # Windows
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
            print("Processing frame...")

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"✅ Saved labeled video: {self.output_video_path}")

# Example usage
if __name__ == "__main__":
    recognizer = CowFaceRecognizer(
        train_dir="/mnt/c/Users/eeuma/Desktop/cow_facial_recognition_yolo_imagenet/dataset_for_classification/train",
        model_path="/mnt/c/Users/eeuma/Desktop/cow_facial_recognition_yolo_imagenet/cow_facial_recognition_efficientnet.pth",
        yolo_model_path="/mnt/c/Users/eeuma/Desktop/cow_facial_recognition_yolo_imagenet/models/fine_tuned/yolov8_fine_tuned_cow_face.pt",
        input_video_path="/mnt/c/Users/eeuma/Desktop/cow_facial_recognition_yolo_imagenet/cow_faces_videos/Cow (10).mp4",
        output_video_path="/mnt/c/Users/eeuma/Desktop/cow_facial_recognition_yolo_imagenet/results/labeled_output10.mp4"
    )
    recognizer.process_video()