import numpy as np
import os

import Utility.custom_model_3d_cnn_utility as CustomModel3DCNNUtils
import Utility.custom_model_utility as CustomModelUtils
import Utility.inceptionv3_utility as InceptionV3Utils
import Utility.resnet50_utility as ResNet50Utils
import Utility.vgg16_utility as VGG16_1_Utils
import Utility.vgg16_2_utility as VGG16_2_Utils


MOVIES_FRAMES_FOLDER_2 = '../2_Movies_Frames/No_Augmentation/'
MOVIES_FRAMES_FOLDER_10 = '../Movies_Frames/No_Augmentation/'
TEST_MOVIE_FOLDER = '../Test_Movies/'
MOVIES_NAME_LIST_1 = [
    'Black.Panther-2018',
    'Bohemian.Rhapsody-2018',
    'Deadpool-2016',
    'Dune.Part.One-2021',
    'Joker-2019',
]

MOVIES_NAME_LIST_2 = [
    'Soul-2020',
    'Spider.Man.Into.the.Spider.Verse-2018',
    'The.Lego.Movie-2014',
    'The.Shawshank.Redemption-1994',
    'Whiplash-2014',
]
ONLY_2_MOVIES_NAME_LIST = [
    'The.Lego.Movie-2014',
    'Whiplash-2014',
]

TMP_FOLDER = 'TMP/'
CUSTOM_MODEL_3D_CNN_FOLDER = TMP_FOLDER + 'CustomModel_3D-CNN/'
CUSTOM_MODEL_FOLDER = TMP_FOLDER + 'CustomModel/'
INCEPTION_V3_FOLDER = TMP_FOLDER + 'InceptionV3/'
RESNET50_FOLDER = TMP_FOLDER + 'ResNet50/'
VGG16_1_FOLDER = TMP_FOLDER + 'VGG16_1/'
VGG16_2_FOLDER = TMP_FOLDER + 'VGG16_2/'


MOVIES_2_LABELS = []
for movie_name in os.listdir(MOVIES_FRAMES_FOLDER_2):
    MOVIES_2_LABELS.append(movie_name)
MOVIES_2_LABELS = np.array(MOVIES_2_LABELS)
MOVIES_2_LABELS = np.unique(MOVIES_2_LABELS)
MOVIES_2_LABELS = np.sort(MOVIES_2_LABELS)

MOVIES_10_LABELS = []
for movie_name in os.listdir(MOVIES_FRAMES_FOLDER_10):
    MOVIES_10_LABELS.append(movie_name)
MOVIES_10_LABELS = np.array(MOVIES_10_LABELS)
MOVIES_10_LABELS = np.unique(MOVIES_10_LABELS)
MOVIES_10_LABELS = np.sort(MOVIES_10_LABELS)


if __name__ == '__main__':
    # Testing CustomModel3DCNN - with only 2 movies in dataset
    print("Custom Model 3D-CNN (trained on 2 movies) predictions:")
    for movie_name in ONLY_2_MOVIES_NAME_LIST:
        tst_movies_samples = CustomModel3DCNNUtils.get_test_movie_samples(movie_name, TEST_MOVIE_FOLDER)
        tst_dataset = CustomModel3DCNNUtils.test_dataset_creation(tst_movies_samples)
        model = CustomModel3DCNNUtils.create_model(MOVIES_FRAMES_FOLDER_2)
        model.load_weights(CUSTOM_MODEL_3D_CNN_FOLDER + 'custom_model_3d_cnn_latest-64img-2movies.h5')

        predictions = model.predict(tst_dataset)
        predicted_labels = np.argmax(predictions, axis=1)
        predicted_labels = [MOVIES_2_LABELS[i] for i in predicted_labels]
        predicted_movie = max(set(predicted_labels), key=predicted_labels.count)
        print('-> Actual movie: ' + movie_name + ' -> Predicted movie: ' + predicted_movie)

    # Testing CustomModel3DCNN - with only 10 movies in dataset
    print("Custom Model 3D-CNN (trained on 10 movies) predictions:")
    for movie_name in MOVIES_NAME_LIST_1:
        tst_movies_samples = CustomModel3DCNNUtils.get_test_movie_samples(movie_name, TEST_MOVIE_FOLDER)
        tst_dataset = CustomModel3DCNNUtils.test_dataset_creation(tst_movies_samples)
        model = CustomModel3DCNNUtils.create_model(MOVIES_FRAMES_FOLDER_10)
        model.load_weights(CUSTOM_MODEL_3D_CNN_FOLDER + 'custom_model_3d_cnn_latest-64img-10movies.h5')

        predictions = model.predict(tst_dataset)
        predicted_labels = np.argmax(predictions, axis=1)
        predicted_labels = [MOVIES_10_LABELS[i] for i in predicted_labels]
        predicted_movie = max(set(predicted_labels), key=predicted_labels.count)
        print('-> Actual movie: ' + movie_name + ' -> Predicted movie: ' + predicted_movie)

    # Testing CustomModel
    print("Custom Model predictions:")
    for movie_name in MOVIES_NAME_LIST_1:
        tst_dataset = CustomModelUtils.create_dataset_from_input_movie(movie_name, TEST_MOVIE_FOLDER)
        model = CustomModelUtils.create_model('../CSV_Movies_Files_Test/Train_Movies.csv')
        CustomModelUtils.compile_model(model)
        model.load_weights(CUSTOM_MODEL_FOLDER + 'custom_model_latest_10movies.hdf5')

        predictions = model.predict(tst_dataset)
        predicted_labels = np.argmax(predictions, axis=1)
        predicted_labels = [MOVIES_10_LABELS[i] for i in predicted_labels]
        predicted_movie = max(set(predicted_labels), key=predicted_labels.count)
        print('-> Actual movie: ' + movie_name + ' -> Predicted movie: ' + predicted_movie)

    # Testing InceptionV3
    print("InceptionV3 predictions:")
    for movie_name in MOVIES_NAME_LIST_1:
        tst_dataset = InceptionV3Utils.create_dataset_from_input_movie(movie_name, TEST_MOVIE_FOLDER)
        model = InceptionV3Utils.create_model('../CSV_Movies_Files_Test/Train_Movies.csv')
        model.load_weights(INCEPTION_V3_FOLDER + 'inceptionv3_latest_10movies.hdf5')

        predictions = model.predict(tst_dataset)
        predicted_labels = np.argmax(predictions, axis=1)
        predicted_labels = [MOVIES_10_LABELS[i] for i in predicted_labels]
        predicted_movie = max(set(predicted_labels), key=predicted_labels.count)
        print('-> Actual movie: ' + movie_name + ' -> Predicted movie: ' + predicted_movie)

    # Testing ResNet50
    print("ResNet50 predictions:")
    for movie_name in MOVIES_NAME_LIST_1:
        tst_dataset = ResNet50Utils.create_dataset_from_input_movie(movie_name, TEST_MOVIE_FOLDER)
        model = ResNet50Utils.create_model('../CSV_Movies_Files/Train_Movies.csv')
        model.load_weights(RESNET50_FOLDER + 'resnet50_latest_10movies.hdf5')

        predictions = model.predict(tst_dataset)
        predicted_labels = np.argmax(predictions, axis=1)
        predicted_labels = [MOVIES_10_LABELS[i] for i in predicted_labels]
        predicted_movie = max(set(predicted_labels), key=predicted_labels.count)
        print('-> Actual movie: ' + movie_name + ' -> Predicted movie: ' + predicted_movie)

    # Testing VGG16_1
    print("VGG16_1 predictions:")
    for movie_name in MOVIES_NAME_LIST_1:
        tst_dataset = VGG16_1_Utils.create_dataset_from_input_movie(movie_name, TEST_MOVIE_FOLDER)
        model = VGG16_1_Utils.create_model('../CSV_Movies_Files_Test/Train_Movies.csv')
        model.load_weights(VGG16_1_FOLDER + 'vgg16_1_latest_10movies.hdf5')

        predictions = model.predict(tst_dataset)
        predicted_labels = np.argmax(predictions, axis=1)
        predicted_labels = [MOVIES_10_LABELS[i] for i in predicted_labels]
        predicted_movie = max(set(predicted_labels), key=predicted_labels.count)
        print('Actual movie: ' + movie_name + ' -> Predicted movie: ' + predicted_movie)

    # Testing VGG16_2
    print("VGG16_2 predictions:")
    for movie_name in MOVIES_NAME_LIST_1:
        tst_dataset = VGG16_2_Utils.create_dataset_from_input_movie(movie_name, TEST_MOVIE_FOLDER)
        model = VGG16_2_Utils.create_model('../CSV_Movies_Files/Train_Movies.csv')
        model.load_weights(VGG16_2_FOLDER + 'vgg16_2_latest_10movies.hdf5')

        predictions = model.predict(tst_dataset)
        predicted_labels = np.argmax(predictions, axis=1)
        predicted_labels = [MOVIES_10_LABELS[i] for i in predicted_labels]
        predicted_movie = max(set(predicted_labels), key=predicted_labels.count)
        print('-> Actual movie: ' + movie_name + ' -> Predicted movie: ' + predicted_movie)
