from ultralytics import YOLO

# load custom dataset from computer

# Step 2: Train YOLOv8 model
# Load YOLOv8 model (select YOLOv8n, YOLOv8s, YOLOv8m, etc., depending on your hardware)
model = YOLO("yolov8s.pt")  # You can change to other YOLOv8 models as needed

# Train the model
results = model.train(
    data=f"data.yaml",  # Path to data.yaml from Roboflow
    epochs=50,  # Adjust epochs as needed
    imgsz=640,  # Image size
    batch=16,  # Adjust based on GPU capacity
    name="cow_face_detection",  # Experiment name
    project="YOLOv8_Projects",  # Output directory
)

# Step 3: Evaluate the model
metrics = model.val()

# Step 4: Predict on new images
predictions = model.predict("test/images/frame_0001_jpg.rf.4c74196bc1e28ac86934f1d01e375400.jpg", conf=0.5)  # Adjust confidence threshold as needed

# Step 5: Export the model (optional)
model.export(format="onnx")  # Export to ONNX for deployment
