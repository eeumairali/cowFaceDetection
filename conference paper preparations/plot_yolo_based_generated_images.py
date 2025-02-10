import glob
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Directory containing the images
image_dir = "/home/cow_facial_recognition_yolo_imagenet/yolo_generated_frames"

# Collect all images (assuming .jpg or .png; adjust as necessary)
image_files = sorted(glob.glob(os.path.join(image_dir, "*.jpg"))) \
            + sorted(glob.glob(os.path.join(image_dir, "*.png")))

# We expect exactly 18 images
if len(image_files) != 18:
    raise ValueError("Expected exactly 18 images, found {}".format(len(image_files)))

# Define grid dimensions
rows = 3
cols = 6

# Create the figure and axes
fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(8,8))  # (width=8, height=10)

# Remove gaps between subplots
plt.subplots_adjust(wspace=0.05, hspace=0.05)

# Iterate over each image and subplot axis
for ax, img_file in zip(axes.flat, image_files):
    # Read the image
    img = mpimg.imread(img_file)
    # Show image on the axes
    ax.imshow(img)
    ax.axis('off')  # Hide the axis ticks

# Hide any remaining empty axes if fewer images
for ax in axes.flat[len(image_files):]:
    ax.set_visible(False)

# Save the figure if needed
plt.savefig("yolov8_high_confidence_images.png", dpi=600, bbox_inches='tight')

# Display the final plot
plt.show()
