import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

MOVIES_FOLDER_PATH = '../Movies_MP4/'


def plot_frames(input_dir):
    # movies_displayed_name = ['Film 1', 'Film 2', 'Film 3', 'Film 4', 'Film 5', 'Film 6', 'Film 7', 'Film 8', 'Film 9', 'Film 10']
    movie_names = []
    frame_counts = []
    for root, dirs, files in os.walk(input_dir):
        for video in files:
            movie_names.append(os.path.splitext(video)[0])
            video_path = os.path.join(root, video)
            cv = cv2.VideoCapture(video_path)
            fps = int(cv.get(cv2.CAP_PROP_FPS))
            total_frames = int(cv.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_interval = 2  # Extract one frame every 2 seconds

            extracted_frames = total_frames // (fps * frame_interval)
            frame_counts.append(extracted_frames)

    reference_frames = [3077] * len(movie_names)

    bar_width = 0.35
    x_positions = range(len(movie_names))

    plt.figure(figsize=(12, 6))

    plt.bar(x_positions, reference_frames, width=bar_width, label='Numărul de cadre extrase', color='green')
    plt.bar([pos + bar_width for pos in x_positions], frame_counts, width=bar_width, label='Numărul actual de cadre', color='red')

    plt.ylabel("Numărul de cadre")
    # plt.title("Numărul total de cadre la fiecare 2 secunde ale fiecărui film vs numărul de cadre extrase")

    plt.xticks([pos + bar_width / 2 for pos in x_positions], movie_names, rotation=45)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    plot_frames(MOVIES_FOLDER_PATH)
