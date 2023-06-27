import cv2
import os
import numpy as np

MOVIES_FOLDER_PATH = '../Movies_MP4/'
MOVIES_FRAMES_FOLDER_PATH = '../Movies_Frames_Test/No_Augmentation/'


def get_min_frames(input_dir):
    min_frames = float('inf')
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            video_path = os.path.join(root, file)
            video = cv2.VideoCapture(video_path)
            frame_rate = video.get(cv2.CAP_PROP_FPS)
            total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_interval = int(2 * frame_rate)
            total_frames = total_frames // frame_interval
            if total_frames < min_frames:
                min_frames = total_frames
    return min_frames


def extract_frames(video_path, output_dir, num_frames):
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, frame_idx in enumerate(frame_indices):
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = video.read()
        if ret:
            new_height = 224
            new_width = 224
            resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
            frame_name = os.path.join(output_dir, f"{i}.jpg")
            cv2.imwrite(frame_name, resized_frame)


def process_videos(input_dir, output_root):
    num_frames = get_min_frames(input_dir)
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            video_path = os.path.join(root, file)
            output_dir = os.path.join(output_root, os.path.splitext(file)[0])
            extract_frames(video_path, output_dir, num_frames)


def movies_extraction():
    process_videos(MOVIES_FOLDER_PATH, MOVIES_FRAMES_FOLDER_PATH)
