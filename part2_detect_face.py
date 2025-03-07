import cv2
import os
from ultralytics import YOLO

class YOLOProcessor:
    def __init__(self, model_path):
        """
        Initialize the YOLO model.
        :param model_path: Path to the trained YOLO model.
        """
        self.model = YOLO(model_path)

    def process_image(self, image_path, output_folder="detected_face_yolov8"):
        """
        Process an image with YOLO and save the result.
        """
        os.makedirs(output_folder, exist_ok=True)
        results = self.model(image_path)  # Run YOLO detection
        for r in results:
            r.save(filename=os.path.join(output_folder, "part2_demo_out.png"))
        print(f"Processed image saved at {output_folder}/part2_demo_out.png")

    def process_video(self, video_path, output_folder="detected_face_yolov8"):
        """
        Process a video frame-by-frame with YOLO and save the result.
        """
        os.makedirs(output_folder, exist_ok=True)
        
        # Load video
        cap = cv2.VideoCapture(video_path)
        frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        output_video_path = os.path.join(output_folder, "part2_demo_out.mp4")
        out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run YOLO detection
            results = self.model(frame)
            for r in results:
                processed_frame = r.plot()  # Get processed frame with bounding boxes

            out.write(processed_frame)  # Save to output video

        cap.release()
        out.release()
        print(f"Processed video saved at {output_video_path}")

# Example Usage
if __name__ == "__main__":
    model_path = "/mnt/c/Users/eeuma/Desktop/cow_facial_recognition_yolo_imagenet/models/fine_tuned/yolov8_fine_tuned_cow_face.pt"
    processor = YOLOProcessor(model_path)

    # Process an image
    image_path = "/mnt/c/Users/eeuma/Desktop/cow_facial_recognition_yolo_imagenet/detected_face_yolov8/part2_demo.png"
    processor.process_image(image_path)

    # Process a video
    video_path = "/mnt/c/Users/eeuma/Desktop/cow_facial_recognition_yolo_imagenet/detected_face_yolov8/part2_demo_video.mp4"
    processor.process_video(video_path)
