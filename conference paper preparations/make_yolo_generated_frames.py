import os
import glob
import cv2
from ultralytics import YOLO

# Path to the directory containing videos
video_dir = "/home/cow_facial_recognition_yolo_imagenet/train/videos"
# Directory to save the extracted best frames
output_dir = "yolo_generated_frames"
os.makedirs(output_dir, exist_ok=True)

# Load the trained YOLOv8 model
model = YOLO("YOLOv8_Projects/cow_face_detection/weights/best.pt")  # Update this path if necessary

# Get all video files from the directory
video_paths = glob.glob(os.path.join(video_dir, "*"))
if not video_paths:
    print(f"No videos found in directory: {video_dir}")
    exit()

for video_path in video_paths:
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        continue

    highest_confidence = -1
    best_frame = None
    frame_count = 0

    print(f"\nProcessing video: {video_name}")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        # Run inference on the current frame
        results = model.predict(source=frame, conf=0.5, imgsz=640, save=False, verbose=False)

        # Check for detections
        if len(results) > 0 and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                cls_idx = int(box.cls)
                class_name = results[0].names[cls_idx]
                confidence = float(box.conf)

                # Since we now know the class name is "cows"
                if class_name.lower() == "cows" and confidence > highest_confidence:
                    highest_confidence = confidence
                    annotated_frame = results[0].plot()
                    best_frame = annotated_frame.copy()

    cap.release()

    # After processing all frames in the video, save the best frame if found
    if best_frame is not None:
        output_path = os.path.join(output_dir, f"{video_name}_best_frame.jpg")
        cv2.imwrite(output_path, best_frame)
        print(f"Highest confidence 'cows' detection for {video_name}: {highest_confidence}")
        print(f"Saved best frame to: {output_path}")
    else:
        print(f"No 'cows' detections found in video: {video_name}")
