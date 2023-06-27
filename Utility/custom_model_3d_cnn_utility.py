import tensorflow as tf
import keras
import numpy as np
import os
from keras import layers
from keras.callbacks import ModelCheckpoint
from . import general as GeneralUtils


MOVIES_FRAMES_FOLDER = '../../../Movies_Frames/No_Augmentation/'

EPOCHS = 25
BATCH_SIZE = 1
BUFFER_SIZE = 256
IMG_HEIGHT = 64
IMG_WIDTH = 64
VIDEO_SAMPLES_LENGTH = 20
INPUT_SHAPE = (VIDEO_SAMPLES_LENGTH, IMG_HEIGHT, IMG_WIDTH, 3)

TRAIN_DATASET_SIZE = 0.9
VAL_DATASET_SIZE = 0.1


def get_movies_labels(movies_folder=MOVIES_FRAMES_FOLDER):
    """Returns the list of the movies in the given folder"""
    MOVIES_LABELS = []
    for movie_name in os.listdir(movies_folder):
        MOVIES_LABELS.append(movie_name)
    MOVIES_LABELS = np.array(MOVIES_LABELS)
    MOVIES_LABELS = np.unique(MOVIES_LABELS)
    MOVIES_LABELS = np.sort(MOVIES_LABELS)
    return MOVIES_LABELS


def get_train_movie_samples_and_labels(movies_folder=MOVIES_FRAMES_FOLDER):
    """Returns the samples and the labels of the training set"""
    MOVIES_LABELS = get_movies_labels()
    MOVIES_SAMPLES = []
    ACTUAL_LABELS = []
    for movie_name in os.listdir(movies_folder):
        movie_dir = movies_folder + movie_name
        frames = os.listdir(movie_dir)
        frames = sorted(frames, key=lambda x: int(x.split('.')[0]))
        current_sample = []
        for frame_index, frame_name in enumerate(frames):
            frame_path = movie_dir + '/' + frame_name
            current_sample.append(frame_path)
            if len(current_sample) == VIDEO_SAMPLES_LENGTH:
                movie_name = frame_path.split('/')[-2]
                sample_label = MOVIES_LABELS.tolist().index(movie_name)
                ACTUAL_LABELS.append(sample_label)
                MOVIES_SAMPLES.append(current_sample)
                current_sample = []

    MOVIES_SAMPLES = np.array(MOVIES_SAMPLES, dtype=object)
    ACTUAL_LABELS = np.array(ACTUAL_LABELS)
    return MOVIES_SAMPLES, ACTUAL_LABELS


# Resize and normalize method and her wrapper
def resize_and_normalize(video_sample):
    """Resize and normalize video sample"""
    resized_video_sample = []
    for frame in video_sample:
        frame = tf.io.read_file(frame)
        frame = tf.image.decode_jpeg(frame, channels=3)
        frame = tf.image.resize(frame, [IMG_HEIGHT, IMG_WIDTH])
        frame = tf.cast(frame, tf.float32) / 255.0
        resized_video_sample.append(frame)

    resized_video_sample = tf.stack(resized_video_sample)
    return resized_video_sample


def resize_and_normalize_wrapper(video_sample, label):
    """Wrapper for resize and normalize method"""
    resized_sample = tf.py_function(resize_and_normalize, [video_sample], tf.float32)
    return resized_sample, label


# Augmentation methods and their wrappers
def flip_right(video_sample):
    augmented_video_sample = []
    for frame in video_sample:
        augmented_frame = tf.reverse(frame, axis=[1])
        augmented_video_sample.append(augmented_frame)

    augmented_video_sample = tf.stack(augmented_video_sample)
    return augmented_video_sample


def flip_right_wrapper(video_sample, label):
    augmented_video_sample = tf.py_function(flip_right, [video_sample], tf.float32)
    return augmented_video_sample, label


def flip_downward(video_sample):
    augmented_video_sample = []
    for frame in video_sample:
        augmented_frame = tf.reverse(frame, axis=[0])
        augmented_video_sample.append(augmented_frame)

    augmented_video_sample = tf.stack(augmented_video_sample)
    return augmented_video_sample


def flip_downward_wrapper(video_sample, label):
    augmented_video_sample = tf.py_function(flip_downward, [video_sample], tf.float32)
    return augmented_video_sample, label


def brightness(video_sample):
    augmented_video_sample = []
    for frame in video_sample:
        augmented_frame = tf.image.adjust_brightness(frame, 0.4)
        augmented_video_sample.append(augmented_frame)

    augmented_video_sample = tf.stack(augmented_video_sample)
    return augmented_video_sample


def brightness_wrapper(video_sample, label):
    augmented_video_sample = tf.py_function(brightness, [video_sample], tf.float32)
    return augmented_video_sample, label


def saturation(video_sample):
    augmented_video_sample = []
    for frame in video_sample:
        augmented_frame = tf.image.adjust_saturation(frame, 2)
        augmented_video_sample.append(augmented_frame)

    augmented_video_sample = tf.stack(augmented_video_sample)
    return augmented_video_sample


def saturation_wrapper(video_sample, label):
    augmented_video_sample = tf.py_function(saturation, [video_sample], tf.float32)
    return augmented_video_sample, label


def crop(video_sample):
    augmented_video_sample = []
    for frame in video_sample:
        augmented_frame = tf.image.central_crop(frame, 0.9)
        augmented_video_sample.append(augmented_frame)

    augmented_video_sample = tf.stack(augmented_video_sample)
    return augmented_video_sample


def crop_wrapper(video_sample, label):
    augmented_video_sample = tf.py_function(crop, [video_sample], tf.float32)
    return augmented_video_sample, label


def rotate_right_90_degrees(video_sample):
    augmented_video_sample = []
    for frame in video_sample:
        augmented_frame = tf.image.rot90(frame, k=1)
        augmented_video_sample.append(augmented_frame)

    augmented_video_sample = tf.stack(augmented_video_sample)
    return augmented_video_sample


def rotate_right_90_degrees_wrapper(video_sample, label):
    augmented_video_sample = tf.py_function(rotate_right_90_degrees, [video_sample], tf.float32)
    return augmented_video_sample, label


def rotate_left_90_degrees(video_sample):
    augmented_video_sample = []
    for frame in video_sample:
        frame = tf.reverse(frame, axis=[1])
        augmented_frame = tf.image.rot90(frame, k=3)
        augmented_video_sample.append(augmented_frame)

    augmented_video_sample = tf.stack(augmented_video_sample)
    return augmented_video_sample


def rotate_left_90_degrees_wrapper(video_sample, label):
    augmented_video_sample = tf.py_function(rotate_left_90_degrees, [video_sample], tf.float32)
    return augmented_video_sample, label


def dataset_creation():
    """Create the main dataset without the augmentation methods"""
    movies, labels = get_train_movie_samples_and_labels()
    dataset = tf.data.Dataset.from_tensor_slices((movies, labels))
    return dataset


def train_dataset_creation():
    """Create the train dataset with the augmentation methods"""
    dataset = dataset_creation()
    train_size = int(TRAIN_DATASET_SIZE * len(dataset))

    train_dataset = dataset \
        .map(
            lambda video_sample, label: resize_and_normalize_wrapper(video_sample, label),
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        ) \
        .take(train_size) \
        .shuffle(BUFFER_SIZE, seed=21) \
        .batch(BATCH_SIZE) \
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    flipped_right_train_dataset = dataset \
        .map(
            lambda video_sample, label: resize_and_normalize_wrapper(video_sample, label),
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        ) \
        .take(train_size) \
        .shuffle(BUFFER_SIZE, seed=21) \
        .batch(BATCH_SIZE) \
        .map(
            lambda video_sample, label: flip_right_wrapper(video_sample, label),
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        ) \
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    flipped_downward_train_dataset = dataset \
        .map(
            lambda video_sample, label: resize_and_normalize_wrapper(video_sample, label),
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        ) \
        .take(train_size) \
        .shuffle(BUFFER_SIZE, seed=21) \
        .batch(BATCH_SIZE) \
        .map(
            lambda video_sample, label: flip_downward_wrapper(video_sample, label),
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        ) \
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    brightness_train_dataset = dataset \
        .map(
            lambda video_sample, label: resize_and_normalize_wrapper(video_sample, label),
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        ) \
        .take(train_size) \
        .shuffle(BUFFER_SIZE, seed=21) \
        .batch(BATCH_SIZE) \
        .map(
            lambda video_sample, label: brightness_wrapper(video_sample, label),
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        ) \
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    saturation_train_dataset = dataset \
        .map(
            lambda video_sample, label: resize_and_normalize_wrapper(video_sample, label),
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        ) \
        .take(train_size) \
        .shuffle(BUFFER_SIZE, seed=21) \
        .batch(BATCH_SIZE) \
        .map(
            lambda video_sample, label: saturation_wrapper(video_sample, label),
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        ) \
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    crop_train_dataset = dataset \
        .map(
            lambda video_sample, label: resize_and_normalize_wrapper(video_sample, label),
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        ) \
        .take(train_size) \
        .shuffle(BUFFER_SIZE, seed=21) \
        .batch(BATCH_SIZE) \
        .map(
            lambda video_sample, label: crop_wrapper(video_sample, label),
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        ) \
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    train_dataset = train_dataset.concatenate(flipped_right_train_dataset)
    train_dataset = train_dataset.concatenate(flipped_downward_train_dataset)
    train_dataset = train_dataset.concatenate(brightness_train_dataset)
    train_dataset = train_dataset.concatenate(saturation_train_dataset)
    # train_dataset = train_dataset.concatenate(crop_train_dataset)

    return train_dataset


def validation_dataset_creation():
    """Create the validation dataset"""
    dataset = dataset_creation()
    train_size = int(TRAIN_DATASET_SIZE * len(dataset))

    val_dataset = dataset \
        .map(
            lambda video_sample, label: resize_and_normalize_wrapper(video_sample, label),
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        ) \
        .skip(train_size) \
        .batch(BATCH_SIZE) \
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return val_dataset


def get_test_movie_samples(movie_name, test_movie_folder='../Movies_Frames/Test/'):
    """Get the test movie samples"""
    test_movies_samples = []
    frames = os.listdir(test_movie_folder + movie_name)
    frames = sorted(frames, key=lambda x: int(x.split('.')[0]))
    current_sample = []
    for frame_index, frame_name in enumerate(frames):
        frame_path = test_movie_folder + movie_name + '/' + frame_name
        current_sample.append(frame_path)
        if len(current_sample) == VIDEO_SAMPLES_LENGTH:
            test_movies_samples.append(current_sample)
            current_sample = []

    test_movies_samples = np.array(test_movies_samples, dtype=object)
    return test_movies_samples


def resize_and_normalize_test_dataset_wrapper(video_sample):
    """Resize and normalize a video sample from the test dataset wrapper function"""
    resized_sample = tf.py_function(resize_and_normalize, [video_sample], tf.float32)
    return resized_sample


def test_dataset_creation(test_movie_samples):
    """Create a test dataset from a list of movie samples"""
    test_dataset = tf.data.Dataset.from_tensor_slices(test_movie_samples)
    test_dataset = test_dataset.map(
            lambda video_sample: resize_and_normalize_test_dataset_wrapper(video_sample),
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        ) \
        .batch(BATCH_SIZE) \
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return test_dataset


def create_model(movie_folder=MOVIES_FRAMES_FOLDER):
    """Create a 3D CNN model"""
    MOVIES_LABELS = get_movies_labels(movie_folder)
    model = keras.Sequential([
        layers.Conv3D(16, (1, 3, 3), activation='relu', input_shape=INPUT_SHAPE, padding='same'),
        layers.Conv3D(16, (3, 1, 1), activation='relu', padding='same'),
        layers.MaxPooling3D(pool_size=(2, 2, 2)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Conv3D(32, (1, 3, 3), activation='relu', padding='same'),
        layers.Conv3D(32, (3, 1, 1), activation='relu', padding='same'),
        layers.MaxPooling3D(pool_size=(2, 2, 2)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.GlobalAveragePooling3D(),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(len(MOVIES_LABELS), activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        metrics=['accuracy']
    )

    return model


def create_checkpoint(date, time):
    """Create a checkpoint to save the best model"""
    SAVED_MODEL_PATH = GeneralUtils.SAVED_MODELS_PATH + 'CustomModel_3D-CNN/'
    filename = 'custom_model_3d_cnn_' + date + '_' + time + '.h5'
    filepath = SAVED_MODEL_PATH + filename
    return ModelCheckpoint(
        filepath=filepath,
        save_best_only=True,
        monitor='accuracy',
        mode='max',
        save_weights_only=True
    )


def create_checkpoint_latest():
    """Create a checkpoint to save the latest model"""
    SAVED_MODEL_PATH = GeneralUtils.SAVED_MODELS_PATH + 'CustomModel_3D-CNN/'
    filepath = SAVED_MODEL_PATH + 'custom_model_3d_cnn_latest_10movies.h5'
    return ModelCheckpoint(
        filepath=filepath,
        save_best_only=True,
        monitor='accuracy',
        mode='max',
        save_weights_only=True
    )


def save_training(tr_time, date, time, acc, loss, val_acc, val_loss):
    """Save the training time in a file"""
    SAVED_MODEL_PATH = GeneralUtils.SAVED_MODELS_PATH + 'CustomModel_3D-CNN/'
    MODEL_TRAIN_TIME_FILE = SAVED_MODEL_PATH + 'train.txt'
    tr_time = int(tr_time)
    minutes, seconds = divmod(tr_time, 60)
    tr_time = str(minutes) + ' minutes - ' + str(seconds) + ' seconds'
    GeneralUtils.save_training(tr_time, date, time, EPOCHS, acc, loss, val_acc, val_loss, MODEL_TRAIN_TIME_FILE)


def save_testing(test_time, date, time, acc):
    """Save the testing time in a file"""
    SAVED_MODEL_PATH = GeneralUtils.SAVED_MODELS_PATH + 'CustomModel_3D-CNN/'
    MODEL_TEST_TIME_FILE = SAVED_MODEL_PATH + 'test.txt'
    test_time = int(test_time)
    minutes, seconds = divmod(test_time, 60)
    test_time = str(minutes) + ' minutes - ' + str(seconds) + ' seconds'
    GeneralUtils.save_testing(test_time, date, time, EPOCHS, acc, MODEL_TEST_TIME_FILE)
