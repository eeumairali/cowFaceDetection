"""
🟩 EfficientNet/ResNet Model Training Script

🟢 **Model:** EfficientNet-B0 (default), can be adapted to ResNet
🟢 **Purpose:** Train a deep neural network to classify individual cow faces.
🟢 **Key Features:**
    - Modern convolutional neural network (CNN) for image classification.
    - EfficientNet: Compound scaling (width, depth, resolution) for optimal accuracy/efficiency.
    - ResNet: Residual connections for very deep networks (e.g., ResNet-50, ResNet-101).
    - Data augmentation: Random rotation, crop, flip, normalization.
    - Customizable number of classes (auto-detected from dataset).
    - Training metrics and logs displayed in Streamlit.

🟢 **Technical Details:**
    - Input size: 224x224 (default, configurable).
    - Convolutional kernel sizes: 3x3, 1x1 (varies by layer).
    - EfficientNet-B0: ~5.3M params, 18 layers, MBConv blocks, Swish activation.
    - ResNet: 34-152 layers, uses skip connections, ReLU activation.
    - Loss: CrossEntropyLoss.
    - Optimizer: Adam (default).
    - Output: Class probabilities for each cow.

🟢 **Usage:**
    - Trains EfficientNet/ResNet on cropped cow face dataset.
    - Saves model weights in a dedicated folder for later inference.

"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import os
import streamlit as st
import datetime

class CowFacialRecognition:
    def __init__(self, dataset_path, img_size=224, batch_size=32, lr=0.001, epochs=10, trained_dir="models/efficientnet_trained"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        st.info(f"Using device: {self.device}")
        self.dataset_path = dataset_path
        self.train_dir = os.path.join(dataset_path, "train")
        self.test_dir = os.path.join(dataset_path, "test")
        self.img_size = img_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.trained_dir = trained_dir
        os.makedirs(self.trained_dir, exist_ok=True)
        self._prepare_data()
        self._build_model()

    def _prepare_data(self):
        transform = {
            "train": transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.RandomRotation(30),
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(self.img_size, scale=(0.8, 1.0)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            "test": transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        }
        self.train_dataset = datasets.ImageFolder(self.train_dir, transform=transform["train"])
        self.test_dataset = datasets.ImageFolder(self.test_dir, transform=transform["test"])
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
        self.num_classes = len(self.train_dataset.classes)

    def _build_model(self):
        self.model = models.efficientnet_b0(pretrained=True)
        self.model.classifier = nn.Sequential(
            nn.Linear(self.model.classifier[1].in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, self.num_classes)
        )
        self.model = self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def train(self):
        with st.spinner("Training ResNet model..."):
            for epoch in range(self.epochs):
                self.model.train()
                running_loss = 0.0
                correct = 0
                total = 0
                for images, labels in self.train_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    self.optimizer.zero_grad()
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()
                    running_loss += loss.item()
                    _, predicted = outputs.max(1)
                    correct += (predicted == labels).sum().item()
                    total += labels.size(0)
                train_acc = 100 * correct / total
                st.info(f"Epoch [{epoch+1}/{self.epochs}], Loss: {running_loss/len(self.train_loader):.4f}, Accuracy: {train_acc:.2f}%")
            self.save_model()
        st.success("🎉 Training Complete! Model saved.")

    def save_model(self, filename=None):
        if filename is None:
            dt_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = os.path.join(self.trained_dir, f"efficientnet_{dt_str}.pth")
        torch.save(self.model.state_dict(), filename)
        st.info(f"Model saved as {filename}")

    @staticmethod
    def list_trained_models(trained_dir="models/efficientnet_trained"):
        if not os.path.exists(trained_dir):
            return []
        return [f for f in os.listdir(trained_dir) if f.endswith('.pth')]
