import os
import shutil

def capture_one_pic_from_each_subfolder(root_folder, destination_folder):
    """
    Copies one picture from each subfolder of the root_folder to the destination_folder.

    Parameters:
        root_folder (str): Path to the main folder containing subfolders with images.
        destination_folder (str): Path to the folder where the selected images will be saved.
    """
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    subfolders = [subfolder for subfolder in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, subfolder))]
    subfolders = subfolders[:36]  # Limit to the first 36 subfolders

    for subfolder in subfolders:
        subfolder_path = os.path.join(root_folder, subfolder)

        # List all files in the subfolder
        files = [f for f in os.listdir(subfolder_path) if os.path.isfile(os.path.join(subfolder_path, f))]

        # Filter for image files (common extensions)
        image_files = [f for f in files if f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif', 'tiff'))]

        if image_files:
            # Select the first image file
            first_image = image_files[0]
            source_path = os.path.join(subfolder_path, first_image)
            dest_path = os.path.join(destination_folder, f"{subfolder}_{first_image}")

            # Copy the image to the destination folder
            shutil.copy(source_path, dest_path)
            print(f"Copied: {source_path} -> {dest_path}")

if __name__ == "__main__":
    root_folder = "/home/cow_facial_recognition_yolo_imagenet/cropped_annotated_images"  # Replace with your root folder path
    destination_folder = "cropped_frames36"  # Replace with your destination folder path

    capture_one_pic_from_each_subfolder(root_folder, destination_folder)
    print("Task completed.")
