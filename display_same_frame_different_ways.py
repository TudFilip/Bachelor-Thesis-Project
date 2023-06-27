import os
import random
from PIL import Image
import matplotlib.pyplot as plt

# Define the main directory path
main_dir = "../Movies_Frames"

# Create a list of all augmentation methods and movie names
aug_methods = [d for d in os.listdir(main_dir) if os.path.isdir(os.path.join(main_dir, d))]
movies = [d for d in os.listdir(os.path.join(main_dir, aug_methods[0])) if os.path.isdir(os.path.join(main_dir, aug_methods[0], d))]

# Randomly select a movie and a frame number
selected_movie = random.choice(movies)
selected_frame_number = random.randint(0, 3076)  # Assuming there are at least 100 frames in each movie

# Initialize a list to store resized frames
resized_frames = []

# Extract and resize the selected frames from all augmentation methods
for aug_method in aug_methods:
    frame_path = os.path.join(main_dir, aug_method, selected_movie, f"{selected_frame_number}.jpg")
    frame = Image.open(frame_path)
    resized_frame = frame.resize((int(frame.width / 4), int(frame.height / 4)), Image.ANTIALIAS)
    resized_frames.append(resized_frame)

# Display the resized frames in a plot
fig, axes = plt.subplots(2, 3, figsize=(11.69 / 2, 8.27 / 2))
for i, ax in enumerate(axes.flat):
    ax.imshow(resized_frames[i])
    ax.set_title(aug_methods[i])
    ax.axis('off')

plt.tight_layout()
plt.show()
