import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

class CowFacialRecognition:
    def __init__(self, train_dir, model_path, device=None, img_size=224):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        print("Using device:", self.device)
        
        self.train_dir = train_dir
        self.model_path = model_path
        self.img_size = img_size
        self.class_names = sorted(os.listdir(self.train_dir))  # Sorted to ensure consistent indexing
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.model = self._load_model()
    
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
        image = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image)
            _, predicted = torch.max(outputs, 1)
            predicted_class = self.class_names[predicted.item()]
        
        return predicted_class

# Usage example
if __name__ == "__main__":
    train_dir = "/mnt/c/Users/eeuma/Desktop/cow_facial_recognition_yolo_imagenet/dataset_for_classification/train"
    model_path = "/mnt/c/Users/eeuma/Desktop/cow_facial_recognition_yolo_imagenet/cow_facial_recognition_efficientnet.pth"
    test_image_path = "/mnt/c/Users/eeuma/Desktop/cow_facial_recognition_yolo_imagenet/dataset_for_classification/test/Cow (9)/27.jpg"
    
    recognizer = CowFacialRecognition(train_dir, model_path)
    predicted_class = recognizer.predict(test_image_path)
    print(f"Predicted Class: {predicted_class}")
