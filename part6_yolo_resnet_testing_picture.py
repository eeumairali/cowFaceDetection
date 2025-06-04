import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageDraw, ImageFont
import os
from ultralytics import YOLO
import streamlit as st
import datetime

class CowFacialRecognition:
    def __init__(self, train_dir, model_path=None, yolo_model_path=None, device=None, trained_dir_efficientnet="models/efficientnet_trained", trained_dir_yolo="models/yolov8_trained"):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        st.info(f"Using device: {self.device}")
        self.class_names = sorted(os.listdir(train_dir))
        self.trained_dir_efficientnet = trained_dir_efficientnet
        self.trained_dir_yolo = trained_dir_yolo
        self.transform = self._get_transform()
        if model_path is None:
            available_models = CowFacialRecognition.list_trained_models(self.trained_dir_efficientnet)
            if not available_models:
                st.error(f"No trained EfficientNet models found in {self.trained_dir_efficientnet}.")
                raise FileNotFoundError(f"No trained EfficientNet models found in {self.trained_dir_efficientnet}.")
            model_path = st.selectbox("Select a trained EfficientNet model", available_models, key="efficientnet_model_select_pic")
            model_path = os.path.join(self.trained_dir_efficientnet, model_path)
        if yolo_model_path is None:
            from part1_yolov8_enhanced_dataset import YOLOTrainer
            available_yolo = YOLOTrainer.list_trained_models(self.trained_dir_yolo)
            if not available_yolo:
                st.error(f"No trained YOLO models found in {self.trained_dir_yolo}.")
                raise FileNotFoundError(f"No trained YOLO models found in {self.trained_dir_yolo}.")
            yolo_model_path = st.selectbox("Select a trained YOLO model", available_yolo, key="yolo_model_select_pic")
            yolo_model_path = os.path.join(self.trained_dir_yolo, yolo_model_path)
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
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "C:/Windows/Fonts/arial.ttf"
        ]
        for font_path in font_paths:
            if os.path.exists(font_path):
                return ImageFont.truetype(font_path, font_size)
        return ImageFont.load_default()

    def predict(self, image_path, output_image_path):
        original_image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(original_image)
        results = self.yolo_model.predict(image_path)
        with st.spinner("Running YOLO+ResNet on image..."):
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
        st.success(f"✅ Saved output image: {output_image_path}")
        st.image(original_image, caption="YOLO+ResNet Output", use_column_width=True)

    @staticmethod
    def list_trained_models(trained_dir="models/efficientnet_trained"):
        if not os.path.exists(trained_dir):
            return []
        return [f for f in os.listdir(trained_dir) if f.endswith('.pth')]