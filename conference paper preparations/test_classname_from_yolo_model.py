from ultralytics import YOLO

# Load your model
model = YOLO("YOLOv8_Projects/cow_face_detection/weights/best.pt")

# Print out the class names
print("Class names used by the model:")
print(model.names)
