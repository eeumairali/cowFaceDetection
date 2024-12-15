import os
import glob
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Set matplotlib font to Times New Roman or a similar IEEE-style font
rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = 10  # Adjust as per IEEE guidelines (often ~8-10 pt for figure text)
rcParams['axes.titleweight'] = 'normal'  # Keep title weight normal
rcParams['axes.titlepad'] = 10           # Some padding can improve readability

# Directories
raw_image_dir = "/home/cow_facial_recognition_yolo_imagenet/frames36"
cropped_image_dir = "/home/cow_facial_recognition_yolo_imagenet/cropped_frames36"  # Adjust to your cropped image path

# Load raw images
raw_image_paths = glob.glob(os.path.join(raw_image_dir, "*.jpg"))
raw_image_paths.sort()

# Load cropped images
cropped_image_paths = glob.glob(os.path.join(cropped_image_dir, "*.jpg"))
cropped_image_paths.sort()

# Ensure both lists have the same length or handle the mismatch
min_length = min(len(raw_image_paths), len(cropped_image_paths))
raw_image_paths = raw_image_paths[:min_length]
cropped_image_paths = cropped_image_paths[:min_length]

# Set the number of rows and columns for each image set
rows, cols = 3, 3
total_plots = rows * cols

# If you have fewer images than total_plots, adjust total_plots
total_plots = min(total_plots, min_length)

# Define the desired size for all images
target_size = (224, 224)

# Create the figure
fig, axes = plt.subplots(rows, cols * 2, figsize=(6, 4))

# Add a main title for the entire figure

# Display images
for i in range(total_plots):
    # Compute the row and column index for the raw image
    row = i // cols
    col = i % cols

    # Open and resize raw image
    raw_img = Image.open(raw_image_paths[i])
    raw_img = raw_img.resize(target_size, Image.LANCZOS)

    # Display raw image
    ax_raw = axes[row, col]
    ax_raw.imshow(raw_img)
    ax_raw.axis('off')

    # Open and resize cropped image
    cropped_img = Image.open(cropped_image_paths[i])
    cropped_img = cropped_img.resize(target_size, Image.LANCZOS)

    # Display cropped image
    ax_cropped = axes[row, col + cols]
    ax_cropped.imshow(cropped_img)
    ax_cropped.axis('off')

# Add separate titles above the raw and cropped columns
axes[0, (cols // 2)].set_title("Raw Cow Images", fontsize=10)
axes[0, (cols + (cols // 2))].set_title("Cropped Cow Images", fontsize=10)

# If there are empty subplots, turn off their axes
if total_plots < rows * cols:
    for i in range(total_plots, rows*cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')
        axes[row, col + cols].axis('off')

# Adjust spacing between subplots
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for the main title

# Save the figure
plt.savefig("cow_raw_vs_cropped.png", bbox_inches='tight', dpi=300)

# Show the plot
plt.show()
