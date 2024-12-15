import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import re

# Parameters
IMG_HEIGHT = 224
IMG_WIDTH = 224
model_save_path = "trained_imagenet_model.pth"
data_directory = "/home/cow_facial_recognition_yolo_imagenet/cropped_annotated_images"

# Define transformations
transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the trained model
base_model = models.resnet50(pretrained=False)
num_features = base_model.fc.in_features
base_model.fc = torch.nn.Sequential(
    torch.nn.Linear(num_features, 256),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.5),
    torch.nn.Linear(256, 37),  # Assuming 37 classes
    torch.nn.Softmax(dim=1)
)
base_model.load_state_dict(torch.load(model_save_path))
base_model.eval()  # Set the model to evaluation mode
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
base_model = base_model.to(device)

# Sorted class names list
class_names = os.listdir(data_directory)
class_names_sorted = sorted(class_names, key=lambda x: int(re.search(r'\d+', x).group()))

# Function to predict image class
def predict_image_class(image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    image = image.to(device)
    with torch.no_grad():
        outputs = base_model(image)
        _, predicted = torch.max(outputs.data, 1)
        return class_names_sorted[predicted.item()]

# Testing all images
results = {}
for class_name in class_names_sorted:
    folder_path = os.path.join(data_directory, class_name)
    images = os.listdir(folder_path)
    correct_count = 0
    for image_name in images:
        image_path = os.path.join(folder_path, image_name)
        predicted_class = predict_image_class(image_path)
        if predicted_class == class_name:
            correct_count += 1
    accuracy = correct_count / len(images)
    results[class_name] = accuracy

# Print results
for class_name, accuracy in results.items():
    print(f"Accuracy for {class_name}: {accuracy:.2%}")
