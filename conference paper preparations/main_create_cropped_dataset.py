import cv2
import os
from ultralytics import YOLO

def process_videos_in_folder(input_folder, output_folder):
    try:
        # Load the trained YOLOv8 model
        model = YOLO("YOLOv8_Projects/cow_face_detection/weights/best.pt")  # Path to your trained weights

        # Ensure the output folder exists
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Process each video in the folder
        for video_name in os.listdir(input_folder):
            video_path = os.path.join(input_folder, video_name)
            if not video_path.lower().endswith(('.mp4', '.avi', '.mov')):
                continue

            print(f"Processing video: {video_name}")

            # Create a folder for cropped annotated images for this video
            video_output_folder = os.path.join(output_folder, os.path.splitext(video_name)[0])
            if not os.path.exists(video_output_folder):
                os.makedirs(video_output_folder)

            # Load the video using OpenCV
            cap = cv2.VideoCapture(video_path)
            frame_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Perform YOLOv8 inference
                results = model.predict(source=frame, conf=0.5, imgsz=640, save=False)

                # Crop and save detected boxes if confidence > 60%
                for i, (box, conf) in enumerate(zip(results[0].boxes.xyxy, results[0].boxes.conf)):
                    if conf > 0.6:  # Check if confidence is greater than 60%
                        x_min, y_min, x_max, y_max = map(int, box)
                        cropped_image = frame[y_min:y_max, x_min:x_max]

                        # Save cropped image
                        cropped_image_path = os.path.join(video_output_folder, f"frame{frame_count}_box{i}.jpg")
                        cv2.imwrite(cropped_image_path, cropped_image)

                frame_count += 1

            cap.release()

        print(f"Processing complete. Cropped images saved in {output_folder}")

    except Exception as e:
        print(f"Error during video processing: {e}")

# Input and output paths
input_folder_path = "/home/cow_facial_recognition_yolo_imagenet/train/videos"  # Path to your input folder containing videos
output_folder_path = "cropped_annotated_images"  # Path to save the cropped annotated images

# Process the videos in the folder
process_videos_in_folder(input_folder_path, output_folder_path)
