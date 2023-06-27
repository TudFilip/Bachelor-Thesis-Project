import tensorflow as tf
from . import general as GeneralUtils

from keras.applications.inception_v3 import InceptionV3
from keras import models
from keras.layers import Dense, Dropout, Flatten
from keras.callbacks import ModelCheckpoint

EPOCHS = 25
BATCH_SIZE = 32
BUFFER_SIZE = 1024
IMG_HEIGHT = 150
IMG_WIDTH = 150

SAVED_MODEL_PATH = GeneralUtils.SAVED_MODELS_PATH + 'InceptionV3/'
MODEL_TRAIN_TIME_FILE = SAVED_MODEL_PATH + 'train.txt'
MODEL_TEST_TIME_FILE = SAVED_MODEL_PATH + 'test.txt'


def load_images_and_labels(img_path, img_label):
    """Load features of images and labels in the training dataset"""
    image = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    image = tf.cast(image, tf.float32) / 255.0

    label = tf.convert_to_tensor(img_label, dtype=tf.int32)

    return image, label


def load_images_for_testing(img_path):
    """Load and preprocess images in the testing dataset"""
    image = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    image = tf.cast(image, tf.float32) / 255.0

    return image


def create_dataset():
    """Create the dataset for training and validation"""
    tr_paths, tr_labels = GeneralUtils.training_frames_and_labels()
    dataset = tf.data.Dataset.from_tensor_slices((tr_paths, tr_labels))
    return dataset


def create_train_dataset():
    """Create the training dataset"""
    train_ds = create_dataset()
    train_size = int(0.9 * len(train_ds))
    train_dataset = train_ds \
        .map(
            load_images_and_labels,
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        ) \
        .take(train_size) \
        .shuffle(BUFFER_SIZE, seed=21) \
        .batch(BATCH_SIZE) \
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return train_dataset


def create_val_dataset():
    """Create the validation dataset"""
    train_ds = create_dataset()
    train_size = int(0.9 * len(train_ds))
    val_dataset = train_ds \
        .map(
            load_images_and_labels,
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        ) \
        .skip(train_size) \
        .batch(BATCH_SIZE) \
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return val_dataset


def create_test_dataset():
    """Create the testing dataset"""
    test_paths, _ = GeneralUtils.test_frames_and_labels()
    test_ds = tf.data.Dataset.from_tensor_slices(test_paths)
    test_ds = test_ds.map(load_images_for_testing, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_dataset = test_ds \
        .batch(BATCH_SIZE) \
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return test_dataset


def create_dataset_from_input_movie(movie_name, movie_folder_path='../Movies_Frames/Test/'):
    """Create the testing dataset from an input movie"""
    movie_folder = movie_folder_path + movie_name
    movie_frames = GeneralUtils.get_frames_from_folder(movie_folder)
    test_dataset = tf.data.Dataset.from_tensor_slices(movie_frames)
    test_dataset = test_dataset.map(
            load_images_for_testing,
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        ) \
        .batch(BATCH_SIZE) \
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return test_dataset


def create_model(train_movies_frames_csv='../../../CSV_Movies_Files_Test/Train_Movies.csv'):
    """Create the InceptionV3 model"""
    MOVIES = GeneralUtils.get_movies_count(train_movies_frames_csv)
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    for layer in base_model.layers:
        layer.trainable = False

    x = Flatten()(base_model.output)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(MOVIES, activation='softmax')(x)

    model = models.Model(inputs=base_model.input, outputs=x)

    model.build(input_shape=(None, IMG_HEIGHT, IMG_WIDTH, 3))

    model.compile(
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001),
        metrics=['accuracy']
    )

    return model


def create_checkpoint_latest():
    """Create a checkpoint to save the latest model"""
    filepath = SAVED_MODEL_PATH + 'inceptionv3_latest2_10movies.hdf5'
    return ModelCheckpoint(
        filepath=filepath,
        save_best_only=True,
        monitor='accuracy',
        mode='max',
        save_weights_only=True
    )


def create_checkpoint(date, time):
    """Create a checkpoint to save the best model"""
    filename = 'inceptionv3_10movies_' + str(date) + '_' + str(time) + '.hdf5'
    filepath = SAVED_MODEL_PATH + filename
    return ModelCheckpoint(
        filepath=filepath,
        save_best_only=True,
        monitor='accuracy',
        mode='max',
        save_weights_only=True
    )


def save_training(tr_time, date, time, acc, loss, val_acc, val_loss):
    """Save the training time in a file"""
    tr_time = int(tr_time)
    minutes, seconds = divmod(tr_time, 60)
    tr_time = str(minutes) + ' minutes - ' + str(seconds) + ' seconds'
    GeneralUtils.save_training(tr_time, date, time, EPOCHS, acc, loss, val_acc, val_loss, MODEL_TRAIN_TIME_FILE)


def save_testing(test_time, date, time, acc):
    """Save the testing time in a file"""
    test_time = int(test_time)
    minutes, seconds = divmod(test_time, 60)
    test_time = str(minutes) + ' minutes - ' + str(seconds) + ' seconds'
    GeneralUtils.save_testing(test_time, date, time, EPOCHS, acc, MODEL_TEST_TIME_FILE)

