import os
import cv2


class FrameExtractor:
    def __init__(self, input_dir, output_dir, default_frame_interval=3):
        """
        Initialize the FrameExtractor.

        :param input_dir: Path to the directory containing video files.
        :param output_dir: Path to the directory where extracted frames will be saved.
        :param default_frame_interval: Default interval at which frames are extracted (default is every 3rd frame).
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.default_frame_interval = default_frame_interval

    def set_frame_interval(self, frame_interval):
        """
        Update the default frame interval.

        :param frame_interval: New interval for frame extraction.
        """
        self.default_frame_interval = frame_interval
        print(f"Default frame interval set to {frame_interval}.")

    def extract_frames(self, frame_interval=None):
        """
        Extract frames from all videos in the input directory.

        :param frame_interval: Optional custom interval for this extraction run. Defaults to the instance's default.
        """
        frame_interval = frame_interval or self.default_frame_interval
        self._ensure_directory(self.output_dir)

        video_count = 1  # Start sequential folder naming
        for video_file in os.listdir(self.input_dir):
            if self._is_video_file(video_file):
                folder_name = f"cow{video_count:02d}"
                self._process_video(video_file, frame_interval, folder_name)
                video_count += 1
            else:
                print(f"Skipping {video_file}: Not a video file.")

        print("Frame extraction complete!")

    def _process_video(self, video_file, frame_interval, folder_name):
        """
        Process a single video file to extract frames.

        :param video_file: Name of the video file.
        :param frame_interval: Interval at which frames are extracted.
        :param folder_name: Name of the folder to save frames.
        """
        video_path = os.path.join(self.input_dir, video_file)
        video_output_dir = os.path.join(self.output_dir, folder_name)
        self._ensure_directory(video_output_dir)

        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        saved_frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                frame_file = os.path.join(video_output_dir, f'frame_{saved_frame_count:04d}.jpg')
                cv2.imwrite(frame_file, frame)
                saved_frame_count += 1

            frame_count += 1

        cap.release()
        print(f"Processed {video_file}: Extracted {saved_frame_count} frames to {folder_name} at interval {frame_interval}.")

    @staticmethod
    def _ensure_directory(directory):
        """
        Ensure that a directory exists; create it if it doesn't.

        :param directory: Path to the directory.
        """
        os.makedirs(directory, exist_ok=True)

    @staticmethod
    def _is_video_file(file_name):
        """
        Check if a file is a video file based on its extension.

        :param file_name: Name of the file.
        :return: True if the file is a video file; False otherwise.
        """
        video_extensions = ('.mp4', '.avi', '.mkv', '.mov', '.wmv')
        return file_name.lower().endswith(video_extensions)


if __name__ == '__main1__':
    # Define input and output directories
    input_directory = 'videos'
    output_directory = 'extracted_frames'

    # Create FrameExtractor instance with default interval of 3
    extractor = FrameExtractor(input_directory, output_directory, default_frame_interval=7)


    # Extract frames using sequential folder names like cow01, cow02, etc.
    extractor.extract_frames()
