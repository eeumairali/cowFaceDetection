import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageDraw, ImageFont
import os
from ultralytics import YOLO

class CowFacialRecognition:
    def __init__(self, train_dir, model_path, yolo_model_path, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

        self.class_names = sorted(os.listdir(train_dir))  # Sorted for consistency
        self.transform = self._get_transform()
        self.classifier_model = self._load_classifier_model(model_path)
        self.yolo_model = YOLO(yolo_model_path)

    def _get_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def _load_classifier_model(self, model_path):
        num_classes = len(self.class_names)
        model = models.efficientnet_b0(weights=None)
        model.classifier = nn.Sequential(
            nn.Linear(model.classifier[1].in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model

    def _load_font(self, font_size):
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Linux
            "C:/Windows/Fonts/arial.ttf"  # Windows
        ]
        for font_path in font_paths:
            if os.path.exists(font_path):
                return ImageFont.truetype(font_path, font_size)
        return ImageFont.load_default()

    def predict(self, image_path, output_image_path):
        original_image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(original_image)
        results = self.yolo_model.predict(image_path)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                box_height = y2 - y1
                font_size = max(30, int(box_height * 0.05))
                font = self._load_font(font_size)

                face = original_image.crop((x1, y1, x2, y2))
                face = self.transform(face).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    outputs = self.classifier_model(face)
                    _, predicted = torch.max(outputs, 1)
                    predicted_class = self.class_names[predicted.item()]

                draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=4)
                label_x, label_y = x1, max(0, y1 - font_size - 10)
                draw.text((label_x, label_y), predicted_class, fill="red", font=font)

        original_image.save(output_image_path)
        original_image.show()
        print(f"✅ Saved output image: {output_image_path}")

# Example Usage:
if __name__ == "__main__":
    train_dir = "/mnt/c/Users/eeuma/Desktop/cow_facial_recognition_yolo_imagenet/dataset_for_classification/train"
    model_path = "/mnt/c/Users/eeuma/Desktop/cow_facial_recognition_yolo_imagenet/cow_facial_recognition_efficientnet.pth"
    yolo_model_path = "/mnt/c/Users/eeuma/Desktop/cow_facial_recognition_yolo_imagenet/models/fine_tuned/yolov8_fine_tuned_cow_face.pt"
    test_image_path = "/mnt/c/Users/eeuma/Desktop/cow_facial_recognition_yolo_imagenet/dataset_for_classification/test/Cow (9)/27.jpg"
    output_image_path = "/mnt/c/Users/eeuma/Desktop/cow_facial_recognition_yolo_imagenet/results/yolores.png"
    
    recognizer = CowFacialRecognition(train_dir, model_path, yolo_model_path)
    recognizer.predict(test_image_path, output_image_path)