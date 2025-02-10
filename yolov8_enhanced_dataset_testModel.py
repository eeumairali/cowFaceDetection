from ultralytics import YOLO

# Load the trained model
model = YOLO('/mnt/c/Users/eeuma/Desktop/cow_facial_recognition_yolo_imagenet/models/fine_tuned/yolov8_fine_tuned_cow_face.pt')

# Predict on test images
results = model.predict('/mnt/c/Users/eeuma/Desktop/cow_facial_recognition_yolo_imagenet/results/webBasedimages/multiple_cowsWatching_us.png')

# View results
results[0].show()  # Displays the images with predictions
results[0].save()  # Save images with predictions to a folder
