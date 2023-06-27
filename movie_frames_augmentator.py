import os
import cv2
import tqdm
from tqdm import tqdm
import numpy as np

MOVIES_FRAMES_NOAUG_FOLDER_PATH = '../Movies_Frames_Test/No_Augmentation/'
MOVIES_FRAMES_FOLDER_PATH = '../Movies_Frames_Test/'


def frame_random_rotation(frame, frame_count: int, destination_folder: str):
    angle = np.random.randint(-20, 20)
    rotation_matrix = cv2.getRotationMatrix2D((frame.shape[1] / 2, frame.shape[0] / 2), angle, 1)
    rotated_frame = cv2.warpAffine(frame, rotation_matrix, (frame.shape[1], frame.shape[0]))
    frame_name = os.path.join(destination_folder, f'{frame_count}.jpg')
    cv2.imwrite(frame_name, rotated_frame)


def frame_random_crop(frame, frame_count: int, destination_folder: str):
    x = np.random.randint(0, 224 - 200)
    y = np.random.randint(0, 224 - 200)
    cropped_frame = frame[y:y + 200, x:x + 200]
    frame_name = os.path.join(destination_folder, f'{frame_count}.jpg')
    cv2.imwrite(frame_name, cropped_frame)


def frame_random_brightness(frame, frame_count: int, destination_folder: str):
    alpha = np.random.uniform(0.5, 1.5)
    beta = np.random.randint(0.5, 1.5) * 100
    bright_frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
    frame_name = os.path.join(destination_folder, f'{frame_count}.jpg')
    cv2.imwrite(frame_name, bright_frame)


def frame_random_saturation(frame, frame_count: int, destination_folder: str):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv_frame[:, :, 1] = hsv_frame[:, :, 1] * np.random.uniform(0.5, 1.5)
    hsv_frame[:, :, 1][hsv_frame[:, :, 1] > 255] = 255
    hsv_frame[:, :, 2] = hsv_frame[:, :, 2] * np.random.uniform(0.5, 1.5)
    hsv_frame[:, :, 2][hsv_frame[:, :, 2] > 255] = 255
    rgb_frame = cv2.cvtColor(hsv_frame, cv2.COLOR_HSV2BGR)
    frame_name = os.path.join(destination_folder, f'{frame_count}.jpg')
    cv2.imwrite(frame_name, rgb_frame)


def frame_gausian_blur(frame, frame_count: int, destination_folder: str):
    blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)
    frame_name = os.path.join(destination_folder, f'{frame_count}.jpg')
    cv2.imwrite(frame_name, blurred_frame)


def apply_augmentation():
    movies_frames = [f for f in os.listdir(MOVIES_FRAMES_NOAUG_FOLDER_PATH)]
    for movie in tqdm(movies_frames, desc='Movies'):
        frame_count = 0
        movie_frames = [f for f in os.listdir(os.path.join(MOVIES_FRAMES_NOAUG_FOLDER_PATH, movie))]
        movie_frames = sorted(movie_frames, key=lambda x: int(x.split('.')[0]))
        if not os.path.exists(os.path.join(MOVIES_FRAMES_FOLDER_PATH, 'Rotated', movie)):
            os.makedirs(os.path.join(MOVIES_FRAMES_FOLDER_PATH, 'Rotated', movie))
        if not os.path.exists(os.path.join(MOVIES_FRAMES_FOLDER_PATH, 'Cropped', movie)):
            os.makedirs(os.path.join(MOVIES_FRAMES_FOLDER_PATH, 'Cropped', movie))
        if not os.path.exists(os.path.join(MOVIES_FRAMES_FOLDER_PATH, 'Brightened', movie)):
            os.makedirs(os.path.join(MOVIES_FRAMES_FOLDER_PATH, 'Brightened', movie))
        if not os.path.exists(os.path.join(MOVIES_FRAMES_FOLDER_PATH, 'Saturated', movie)):
            os.makedirs(os.path.join(MOVIES_FRAMES_FOLDER_PATH, 'Saturated', movie))
        if not os.path.exists(os.path.join(MOVIES_FRAMES_FOLDER_PATH, 'Blurred', movie)):
            os.makedirs(os.path.join(MOVIES_FRAMES_FOLDER_PATH, 'Blurred', movie))

        for frame in tqdm(movie_frames, desc='Frames'):
            frame_path = os.path.join(MOVIES_FRAMES_NOAUG_FOLDER_PATH, movie, frame)
            frame = cv2.imread(frame_path)
            frame_random_rotation(frame, frame_count, os.path.join(MOVIES_FRAMES_FOLDER_PATH, 'Rotated', movie))
            frame_random_crop(frame, frame_count, os.path.join(MOVIES_FRAMES_FOLDER_PATH, 'Cropped', movie))
            frame_random_brightness(frame, frame_count, os.path.join(MOVIES_FRAMES_FOLDER_PATH, 'Brightened', movie))
            frame_random_saturation(frame, frame_count, os.path.join(MOVIES_FRAMES_FOLDER_PATH, 'Saturated', movie))
            frame_gausian_blur(frame, frame_count, os.path.join(MOVIES_FRAMES_FOLDER_PATH, 'Blurred', movie))
            frame_count += 1
