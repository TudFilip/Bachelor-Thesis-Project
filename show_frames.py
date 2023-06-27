import os
import random
from PIL import Image
import matplotlib.pyplot as plt

# Define the main directory path
main_dir = "../Movies_Frames"

# Create a list of all subdirectories
subdirs = [os.path.join(main_dir, d1, d2) for d1 in os.listdir(main_dir) if os.path.isdir(os.path.join(main_dir, d1))
           for d2 in os.listdir(os.path.join(main_dir, d1)) if os.path.isdir(os.path.join(main_dir, d1, d2))]

# Initialize a list to store all frame file paths
all_frame_paths = []

# Collect all frame file paths from the subdirectories
for subdir in subdirs:
    frame_files = os.listdir(subdir)
    for frame_file in frame_files:
        frame_path = os.path.join(subdir, frame_file)
        all_frame_paths.append(frame_path)

# Randomly select 24 frame paths
selected_frame_paths = random.sample(all_frame_paths, 30)

# Initialize a list to store resized frames
resized_frames = []

# Extract and resize the selected frames
for frame_path in selected_frame_paths:
    frame = Image.open(frame_path)
    resized_frame = frame.resize((int(frame.width / 2), int(frame.height / 2)), Image.ANTIALIAS)
    resized_frames.append(resized_frame)

# Display the resized frames in a plot
fig, axes = plt.subplots(5, 6, figsize=(11.69 / 2, 8.27 / 2))  # A4 size in inches: (11.69, 8.27)
for i, ax in enumerate(axes.flat):
    ax.imshow(resized_frames[i])
    ax.axis('off')

plt.tight_layout()
plt.show()