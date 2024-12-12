import os
import cv2


class ImageAnnotator:
    def __init__(self, input_dir, output_dir_base):
        """
        Initialize the ImageAnnotator.

        :param input_dir: Path to the directory containing subdirectories of images.
        :param output_dir_base: Base path to save annotated images.
        """
        self.input_dir = input_dir
        self.output_dir_base = output_dir_base

    def annotate_folders(self):
        """
        Annotate images in each folder within the input directory.
        """
        folders = sorted([f for f in os.listdir(self.input_dir) if os.path.isdir(os.path.join(self.input_dir, f))])

        if not folders:
            print("No folders found in the input directory.")
            return

        print(f"Found {len(folders)} folders to annotate.")

        for folder in folders:
            folder_path = os.path.join(self.input_dir, folder)
            output_folder = os.path.join(self.output_dir_base, f"annotated_{folder}")
            annotator = FolderAnnotator(folder_path, output_folder)
            annotator.annotate_images()

        print("All folders have been annotated!")

    @staticmethod
    def _ensure_directory(directory):
        """
        Ensure that a directory exists; create it if it doesn't.

        :param directory: Path to the directory.
        """
        os.makedirs(directory, exist_ok=True)


class FolderAnnotator:
    def __init__(self, input_folder, output_folder):
        """
        Initialize the FolderAnnotator.

        :param input_folder: Path to the folder containing images.
        :param output_folder: Path to save annotated images and notes.
        """
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.annotation_file = os.path.join(self.output_folder, "annotations.txt")
        self._ensure_directory(self.output_folder)

    def annotate_images(self):
        """
        Annotate images in the current folder.
        """
        image_files = [f for f in os.listdir(self.input_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        total_images = len(image_files)

        if not image_files:
            print(f"No images found in {self.input_folder}. Skipping...")
            return

        print(f"Annotating {total_images} images in folder: {os.path.basename(self.input_folder)}")

        for i, image_file in enumerate(image_files):
            remaining = total_images - (i + 1)
            print(f"Processing {image_file} ({remaining} images left in this folder).")

            image_path = os.path.join(self.input_folder, image_file)
            annotated_image_path, note = self._annotate_image(image_path, image_file)
            self._save_annotation(annotated_image_path, note)

        print(f"Finished annotating folder: {os.path.basename(self.input_folder)}")

    def _annotate_image(self, image_path, image_file):
        """
        Display the image and take annotation input.

        :param image_path: Path to the image.
        :param image_file: Name of the image file.
        :return: Tuple of annotated image path and note.
        """
        # Load and display the image
        image = cv2.imread(image_path)
        cv2.imshow("Annotate Image", image)
        cv2.waitKey(1)  # Process the GUI event queue

        # Take annotation input
        note = input(f"Enter annotation for {image_file} (leave blank for 'No annotation'): ").strip()
        if not note:
            note = "No annotation"

        # Save the annotated image with an updated name
        base_name, ext = os.path.splitext(image_file)
        annotated_name = f"{base_name}_annotated_{note.replace(' ', '_')}{ext}"
        annotated_image_path = os.path.join(self.output_folder, annotated_name)
        cv2.imwrite(annotated_image_path, image)

        # Close the display window
        cv2.destroyAllWindows()
        return annotated_image_path, note

    def _save_annotation(self, image_path, note):
        """
        Save the annotation details to a text file.

        :param image_path: Path to the annotated image.
        :param note: Annotation note.
        """
        with open(self.annotation_file, "a") as f:
            f.write(f"{os.path.basename(image_path)}: {note}\n")

    @staticmethod
    def _ensure_directory(directory):
        """
        Ensure that a directory exists; create it if it doesn't.

        :param directory: Path to the directory.
        """
        os.makedirs(directory, exist_ok=True)


if __name__ == "__main__":
    # Base directories
    input_directory = "extracted_frames/cow01"  # Folder containing subfolders like cow_01, cow_02, etc.
    output_base_directory = "annotated_cows"  # Base folder for annotated outputs

    # Create ImageAnnotator instance and start annotation
    annotator = ImageAnnotator(input_directory, output_base_directory)
    annotator.annotate_folders()
