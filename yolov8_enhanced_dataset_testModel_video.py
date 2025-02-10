from ultralytics import YOLO
import cv2

# Load the trained model
model = YOLO('/mnt/c/Users/eeuma/Desktop/cow_facial_recognition_yolo_imagenet/models/fine_tuned/yolov8_fine_tuned_cow_face.pt')

# Open the video
cap = cv2.VideoCapture('/mnt/c/Users/eeuma/Desktop/cow_facial_recognition_yolo_imagenet/cow.mp4')

# Get the video frame width, height, and FPS
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create VideoWriter object
out = cv2.VideoWriter('/mnt/c/Users/eeuma/Desktop/cow_facial_recognition_yolo_imagenet/results/output_video.mp4', 
                      cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

print('Processing video frames...', cap.isOpened())
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Predict on the frame
    results = model.predict(frame)

    # Save the processed frame
    out.write(results[0].plot())

    print('Frame processed and saved')

cap.release()
out.release()
cv2.destroyAllWindows()
