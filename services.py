import os
import numpy as np
import cv2

import Utility.custom_model_3d_cnn_utility as CustomModel3DCNNUtils
import Utility.custom_model_utility as CustomModelUtils
import Utility.inceptionv3_utility as InceptionV3Utils
import Utility.resnet50_utility as ResNet50Utils
import Utility.vgg16_utility as VGG16_1_Utils
import Utility.vgg16_2_utility as VGG16_2_Utils


MOVIES_FRAMES_FOLDER_10 = '../Movies_Frames/No_Augmentation/'
SEQUENCE_FOLDER = 'Sequence/'
SEQUENCE_FRAMES_FOLDER = 'Sequence/Frames/'

TMP_FOLDER = 'TMP/'
CUSTOM_MODEL_3D_CNN_FOLDER = TMP_FOLDER + 'CustomModel_3D-CNN/'
CUSTOM_MODEL_FOLDER = TMP_FOLDER + 'CustomModel/'
INCEPTION_V3_FOLDER = TMP_FOLDER + 'InceptionV3/'
RESNET50_FOLDER = TMP_FOLDER + 'ResNet50/'
VGG16_1_FOLDER = TMP_FOLDER + 'VGG16_1/'
VGG16_2_FOLDER = TMP_FOLDER + 'VGG16_2/'

MOVIES_10_LABELS = []
for movie_name in os.listdir(MOVIES_FRAMES_FOLDER_10):
    MOVIES_10_LABELS.append(movie_name)
MOVIES_10_LABELS = np.array(MOVIES_10_LABELS)
MOVIES_10_LABELS = np.unique(MOVIES_10_LABELS)
MOVIES_10_LABELS = np.sort(MOVIES_10_LABELS)


def extract_frames_from_sequence(sequence_path):
    """Extract frames from a sequence and save them in a folder"""
    video = cv2.VideoCapture(sequence_path)
    frame_rate = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = int(2 * frame_rate)
    num_frames = total_frames // frame_interval
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    if not os.path.exists(SEQUENCE_FRAMES_FOLDER):
        os.makedirs(SEQUENCE_FRAMES_FOLDER)
    else:
        for file_name in os.listdir(SEQUENCE_FRAMES_FOLDER):
            os.remove(os.path.join(SEQUENCE_FRAMES_FOLDER, file_name))

    for i, frame_idx in enumerate(frame_indices):
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = video.read()
        if ret:
            new_height = 224
            new_width = 224
            resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
            frame_name = os.path.join(SEQUENCE_FRAMES_FOLDER, f"{i}.jpg")
            cv2.imwrite(frame_name, resized_frame)

    video.release()


def predict_movie(model, movie_frames):
    """Predict the movie of a sequence of frames"""
    predictions = model.predict(movie_frames)
    predicted_labels = np.argmax(predictions, axis=1)
    predicted_labels = [MOVIES_10_LABELS[i] for i in predicted_labels]
    predicted_movie = max(set(predicted_labels), key=predicted_labels.count)
    return predicted_movie


def custom_model_3d_cnn_prediction():
    """Predict the movie of a sequence of frames using the custom model 3D-CNN"""
    tst_movies_samples = CustomModel3DCNNUtils.get_test_movie_samples(SEQUENCE_FRAMES_FOLDER, '')
    tst_dataset = CustomModel3DCNNUtils.test_dataset_creation(tst_movies_samples)
    model = CustomModel3DCNNUtils.create_model(MOVIES_FRAMES_FOLDER_10)
    model.load_weights(CUSTOM_MODEL_3D_CNN_FOLDER + 'custom_model_3d_cnn_latest-64img-10movies.h5')

    predicted_movie = predict_movie(model, tst_dataset)
    return predicted_movie


def custom_model_2d_cnn_prediction():
    """Predict the movie of a sequence of frames using the custom model 2D-CNN"""
    tst_dataset = CustomModelUtils.create_dataset_from_input_movie(SEQUENCE_FRAMES_FOLDER, '')
    model = CustomModelUtils.create_model('../CSV_Movies_Files_Test/Train_Movies.csv')
    CustomModelUtils.compile_model(model)
    model.load_weights(CUSTOM_MODEL_FOLDER + 'custom_model_latest_10movies.hdf5')

    predicted_movie = predict_movie(model, tst_dataset)
    return predicted_movie


def inceptionv3_prediction():
    """Predict the movie of a sequence of frames using the InceptionV3 model"""
    tst_dataset = InceptionV3Utils.create_dataset_from_input_movie(SEQUENCE_FRAMES_FOLDER, '')
    model = InceptionV3Utils.create_model('../CSV_Movies_Files_Test/Train_Movies.csv')
    model.load_weights(INCEPTION_V3_FOLDER + 'inceptionv3_latest_10movies.hdf5')

    predicted_movie = predict_movie(model, tst_dataset)
    return predicted_movie


def resnet50_prediction():
    """Predict the movie of a sequence of frames using the ResNet50 model"""
    tst_dataset = ResNet50Utils.create_dataset_from_input_movie(SEQUENCE_FRAMES_FOLDER, '')
    model = ResNet50Utils.create_model('../CSV_Movies_Files_Test/Train_Movies.csv')
    model.load_weights(RESNET50_FOLDER + 'resnet50_latest_10movies.hdf5')

    predicted_movie = predict_movie(model, tst_dataset)
    return predicted_movie


def vgg16_model_1_prediction():
    """Predict the movie of a sequence of frames using the VGG16 model 1"""
    tst_dataset = VGG16_1_Utils.create_dataset_from_input_movie(SEQUENCE_FRAMES_FOLDER, '')
    model = VGG16_1_Utils.create_model('../CSV_Movies_Files_Test/Train_Movies.csv')
    model.load_weights(VGG16_1_FOLDER + 'vgg16_1_latest_10movies.hdf5')

    predicted_movie = predict_movie(model, tst_dataset)
    return predicted_movie


def vgg16_model_2_prediction():
    """Predict the movie of a sequence of frames using the VGG16 model 2"""
    tst_dataset = VGG16_2_Utils.create_dataset_from_input_movie(SEQUENCE_FRAMES_FOLDER, '')
    model = VGG16_2_Utils.create_model('../CSV_Movies_Files_Test/Train_Movies.csv')
    model.load_weights(VGG16_2_FOLDER + 'vgg16_2_latest_10movies.hdf5')

    predicted_movie = predict_movie(model, tst_dataset)
    return predicted_movie
