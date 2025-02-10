import os
from ultralytics import YOLO
import shutil

# Define the base project directory
project_dir = "/mnt/c/Users/eeuma/Desktop/cow_facial_recognition_yolo_imagenet"
data_path = os.path.join(project_dir, "ehanced_dataset_yolov8", "data.yaml")  # Path to the dataset YAML file

# Define directories for models and results
fine_tuned_dir = os.path.join(project_dir, "models", "fine_tuned")
results_dir = os.path.join(project_dir, "results")
os.makedirs(fine_tuned_dir, exist_ok=True)  # Ensure the fine-tuned model directory exists
os.makedirs(results_dir, exist_ok=True)    # Ensure the results directory exists

# Pretrained model path
pretrained_model = "yolov8n.pt"  # Default COCO pretrained model (nano version)

# Load the pretrained YOLOv8 model
model = YOLO(pretrained_model)

# Fine-tune the model on the custom dataset
model.train(
    data=data_path,              # Path to the dataset YAML file
    epochs=50,                   # Number of epochs for fine-tuning
    batch=16,                    # Batch size (adjust based on GPU memory)
    imgsz=640,                   # Image size (consistent with preprocessing)
    workers=8,                   # Number of data loading workers
    project=results_dir,         # Save training results in the results directory
    name="fine_tune_coco",       # Experiment name
    optimizer="SGD",             # Optimizer (default is SGD, can switch to Adam)
    patience=10,                 # Early stopping patience
    pretrained=True              # Continue training with pretrained weights
)

# Path to the best model weights after training
best_weights_path = os.path.join(results_dir, "fine_tune_coco", "weights", "best.pt")

# Move the best weights to the fine-tuned model directory
model_path_tuned = os.path.join(fine_tuned_dir, "yolov8_fine_tuned_cow_face.pt")
shutil.copy(best_weights_path, model_path_tuned)

print(f"Fine-tuned model saved at {model_path_tuned}")

# Validate the fine-tuned model on the validation set
metrics = model.val()

# Perform inference on a test image
test_image_path = os.path.join(project_dir, "frame_0000_jpg.rf.149f754f06c3550a5b7617649e082e3c.jpg")  # Replace with your test image path
results = model.predict(
    source=test_image_path,  # Path to the test image
    save=True                # Save predictions
)

print("Inference complete. Check the output folder for predictions.")
