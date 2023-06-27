import os
import pandas as pd
import numpy as np

TRAIN_MOVIES_CSV_FILE_PATH = '../../../CSV_Movies_Files_Test/Train_Movies.csv'
TEST_MOVIES_CSV_FILE_PATH = '../../../CSV_Movies_Files_Test/Test_Movies.csv'
SAVED_MODELS_PATH = '../../Tmp/'


def read_csv(file_path):
    """Reads a csv file and returns a pandas dataframe"""
    return pd.read_csv(file_path, low_memory=False, names=['Frames', 'Movie'])


def retrieve_frames_path_and_labels_from_csv(my_csv, movies_list):
    """Returns the path of the frames and the labels from a given cvs file"""
    csv_frames = []
    csv_labels = []
    for i in range(my_csv.shape[0]):
        csv_frames.append("../../../" + my_csv['Frames'][i])
        csv_labels.append(movies_list.index(my_csv['Movie'][i]))
    return np.array(csv_frames), np.array(csv_labels)


def training_test_frames_and_labels():
    """Returns the path of the frames and the labels of the training and test set"""
    TRAIN_MOVIES_CSV = read_csv(TRAIN_MOVIES_CSV_FILE_PATH)
    TEST_MOVIES_CSV = read_csv(TEST_MOVIES_CSV_FILE_PATH)
    MOVIES_LIST = TRAIN_MOVIES_CSV['Movie'].unique()
    MOVIES_LIST = MOVIES_LIST.tolist()
    tr_paths, tr_labels = retrieve_frames_path_and_labels_from_csv(TRAIN_MOVIES_CSV, MOVIES_LIST)
    test_paths, test_labels = retrieve_frames_path_and_labels_from_csv(TEST_MOVIES_CSV, MOVIES_LIST)
    return tr_paths, tr_labels, test_paths, test_labels


def training_frames_and_labels():
    """Returns the path of the frames and the labels of the training set"""
    TRAIN_MOVIES_CSV = read_csv(TRAIN_MOVIES_CSV_FILE_PATH)
    MOVIES_LIST = TRAIN_MOVIES_CSV['Movie'].unique()
    MOVIES_LIST = MOVIES_LIST.tolist()
    tr_paths, tr_labels = retrieve_frames_path_and_labels_from_csv(TRAIN_MOVIES_CSV, MOVIES_LIST)
    return tr_paths, tr_labels


def test_frames_and_labels():
    """Returns the path of the frames and the labels of the test set"""
    TEST_MOVIES_CSV = read_csv(TEST_MOVIES_CSV_FILE_PATH)
    TRAIN_MOVIES_CSV = read_csv(TRAIN_MOVIES_CSV_FILE_PATH)
    MOVIES_LIST = TRAIN_MOVIES_CSV['Movie'].unique()
    MOVIES_LIST = MOVIES_LIST.tolist()
    test_paths, test_labels = retrieve_frames_path_and_labels_from_csv(TEST_MOVIES_CSV, MOVIES_LIST)
    return test_paths, test_labels


def get_movies_list():
    """Returns the list of the movies"""
    TRAIN_MOVIES_CSV = read_csv(TRAIN_MOVIES_CSV_FILE_PATH)
    MOVIES_LIST = TRAIN_MOVIES_CSV['Movie'].unique()
    MOVIES_LIST = MOVIES_LIST.tolist()
    return MOVIES_LIST


def get_movies_count(csv_file=TRAIN_MOVIES_CSV_FILE_PATH):
    """Returns the list of the movies"""
    TRAIN_MOVIES_CSV = read_csv(csv_file)
    MOVIES_LIST = TRAIN_MOVIES_CSV['Movie'].unique()
    MOVIES_LIST = MOVIES_LIST.tolist()
    return len(MOVIES_LIST)


def get_test_csv():
    """Returns the test csv"""
    return read_csv(TEST_MOVIES_CSV_FILE_PATH)


def get_frames_from_folder(movie_folder):
    """Returns the frames of a given movie from a folder"""
    movie_frames = []
    frames = os.listdir(movie_folder)
    frames = sorted(frames, key=lambda x: int(x.split('.')[0]))
    for frame in frames:
        frame_path = movie_folder + '/' + frame
        movie_frames.append(frame_path)
    return np.array(movie_frames)


def save_training(tr_time, date, time, epochs, acc, loss, val_acc, val_loss, file):
    """Saves the training time in a file"""
    with open(file, 'a') as f:
        f.write('Training Date: ' + date + '\n')
        f.write('Training Starting Hour: ' + time + '\n')
        f.write('Training Duration: ' + tr_time + '\n')
        f.write('Number of Epochs: ' + str(epochs) + '\n')
        f.write('Accuracy: ' + str(acc) + '\n')
        f.write('Loss: ' + str(loss) + '\n')
        f.write('Validation Accuracy: ' + str(val_acc) + '\n')
        f.write('Validation Loss: ' + str(val_loss))
        f.write('\n\n----------------------------------------\n\n')


def save_testing(tr_time, date, time, epochs, acc, file):
    """Saves the testing time in a file"""
    with open(file, 'a') as f:
        f.write('Testing Date: ' + date + '\n')
        f.write('Testing Starting Hour: ' + time + '\n')
        f.write('Testing Duration: ' + tr_time + '\n')
        f.write('Number of Epochs: ' + str(epochs) + '\n')
        f.write('Accuracy: ' + str(acc))
        f.write('\n\n----------------------------------------\n\n')
