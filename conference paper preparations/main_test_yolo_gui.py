import tkinter as tk
from tkinter import filedialog
from ultralytics import YOLO
import cv2
import threading

# Function to process video
def process_video(video_path):
    try:
        # Load the trained YOLOv8 model
        model = YOLO("YOLOv8_Projects/cow_face_detection/weights/best.pt")  # Path to your trained weights
        output_path = "prediction_output/output_video.mp4"  # Path to save the processed video

        # Load the video using OpenCV
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Define video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Process video frame by frame
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Perform YOLOv8 inference
            results = model.predict(source=frame, conf=0.5, imgsz=640, save=False)

            # Annotate the frame
            annotated_frame = results[0].plot()

            # Write the annotated frame to the output video
            out.write(annotated_frame)

        # Release resources
        cap.release()
        out.release()
        print(f"Processed video saved as {output_path}")
        status_label.config(text="Processing complete. Output saved as 'output_video.mp4'")
    except Exception as e:
        print(f"Error during video processing: {e}")
        status_label.config(text="Error during processing. Check console for details.")

# Function to open file dialog and select video
def select_video():
    video_path = filedialog.askopenfilename(
        filetypes=[
            ("All files", "*.*")  # Option to show all files
        ]
    )
    if video_path:
        print(f"Selected video: {video_path}")
        status_label.config(text="Processing video... Please wait.")
        # Run the video processing in a separate thread
        threading.Thread(target=process_video, args=(video_path,)).start()

# Create GUI
root = tk.Tk()
root.title("YOLOv8 Video Processor")
root.geometry("400x200")

# Upload button
upload_button = tk.Button(root, text="Upload Video", command=select_video, font=("Arial", 14))
upload_button.pack(pady=20)

# Status label
status_label = tk.Label(root, text="No video selected", font=("Arial", 12))
status_label.pack(pady=20)

# Run the GUI loop
root.mainloop()

