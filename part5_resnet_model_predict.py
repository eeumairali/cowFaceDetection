import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import streamlit as st
import datetime

class CowFacialRecognition:
    def __init__(self, train_dir, model_path=None, device=None, img_size=224, trained_dir="models/efficientnet_trained"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        st.info(f"Using device: {self.device}")
        self.train_dir = train_dir
        self.trained_dir = trained_dir
        self.img_size = img_size
        self.class_names = sorted(os.listdir(self.train_dir))  # Sorted to ensure consistent indexing
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        # Ensure trained_dir exists
        if not os.path.exists(self.trained_dir):
            os.makedirs(self.trained_dir, exist_ok=True)
        # If no model_path is given, prompt user to select from available models
        available_models = self.list_trained_models(self.trained_dir)
        if not available_models:
            st.error(f"No trained models found in {self.trained_dir}.")
            raise FileNotFoundError(f"No trained models found in {self.trained_dir}.")
        if model_path is None:
            model_path = st.selectbox("Select a trained EfficientNet model", available_models, key="efficientnet_model_select")
            model_path = os.path.join(self.trained_dir, model_path)
        self.model_path = model_path
        self.model = self._load_model()

    @staticmethod
    def list_trained_models(trained_dir="models/efficientnet_trained"):
        if not os.path.exists(trained_dir):
            return []
        return [f for f in os.listdir(trained_dir) if f.endswith('.pth')]

    def _load_model(self):
        num_classes = len(self.class_names)
        model = models.efficientnet_b0(pretrained=False)
        model.classifier = nn.Sequential(
            nn.Linear(model.classifier[1].in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        model = model.to(self.device)
        model.eval()
        return model
    
    def predict(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(image_tensor)
            _, predicted = torch.max(outputs, 1)
            predicted_class = self.class_names[predicted.item()]
        st.success(f"Predicted Class: {predicted_class}")
        st.image(image, caption=f"Predicted: {predicted_class}", use_column_width=True)
        return predicted_class

# Removed CLI usage example for Streamlit integration
