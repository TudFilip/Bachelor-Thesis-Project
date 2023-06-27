import os
import pandas as pd

from sklearn.model_selection import train_test_split

MOVIES_FRAMES_FOLDER_PATH = '../Movies_Frames_Test/'
MOVIES_FOLDER_PATH = '../Movies_MP4/'
CSV_FILE_PATH = '../CSV_Movies_Files_Test/'


def movies_frames_to_csv():
    folders = [f for f in os.listdir(MOVIES_FRAMES_FOLDER_PATH)]
    for folder in folders:
        movies_in_folder = [f for f in os.listdir(os.path.join(MOVIES_FRAMES_FOLDER_PATH, folder))]
        for movie in movies_in_folder:
            frames_name_list = []
            frames = [f for f in os.listdir(os.path.join(MOVIES_FRAMES_FOLDER_PATH, folder, movie))]
            frames = sorted(frames, key=lambda x: int(x.split('.')[0]))
            for frame in frames:
                frames_name_list.append(f'{movie}?{folder}?{frame}')

            df = pd.DataFrame()
            df['Frames'] = frames_name_list
            df['Movie'] = movie

            if os.path.exists(os.path.join(CSV_FILE_PATH, f'{movie}.csv')):
                df.to_csv(os.path.join(CSV_FILE_PATH, f'{movie}.csv'), mode='a', header=False, index=False)
            else:
                df.to_csv(os.path.join(CSV_FILE_PATH, f'{movie}.csv'), index=False)

# 1. prima oara rulez asta sa imi creez csv urile cu frameurile din fiecare film in parte
# movies_frames_to_csv()


def combine_all_csv():
    csv_files = [f for f in os.listdir(CSV_FILE_PATH)]
    df = pd.DataFrame()
    for csv_file in csv_files:
        df = df.append(pd.read_csv(os.path.join(CSV_FILE_PATH, csv_file)), ignore_index=True)

    if os.path.exists(os.path.join(CSV_FILE_PATH, 'All_Movies.csv')):
        os.remove(os.path.join(CSV_FILE_PATH, 'All_Movies.csv'))
    df.to_csv(os.path.join(CSV_FILE_PATH, 'All_Movies.csv'), index=False, header=False)

# 2. rulez asta ca sa combin toate csv-urile in unul singur, astfel incat sa am toate frameurile in acelasi csv
# combine_all_csv()


def split_data_to_train_test():
    df = pd.read_csv(
        os.path.join(CSV_FILE_PATH, 'All_Movies.csv'),
        names=['Frames', 'Movie'],
        low_memory=False
    )

    train_size = 0.9

    X = df.drop(columns=['Movie']).copy()
    y = df['Movie']

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=21, stratify=y)

    train_images = []
    train_classes = []

    for i in range(len(X_train['Frames'])):
        train_images.append('Movies_Frames/' +
                            X_train['Frames'].iloc[i].split('?')[1] +
                            '/' +
                            X_train['Frames'].iloc[i].split('?')[0] +
                            '/' +
                            X_train['Frames'].iloc[i].split('?')[2]
                            )
        train_classes.append(X_train['Frames'].iloc[i].split('?')[0])

    train_df = pd.DataFrame()
    train_df['Frames'] = train_images
    train_df['Movie'] = train_classes
    if os.path.exists(os.path.join(CSV_FILE_PATH, 'Train_Movies.csv')):
        os.remove(os.path.join(CSV_FILE_PATH, 'Train_Movies.csv'))
    train_df.to_csv(os.path.join(CSV_FILE_PATH, 'Train_Movies.csv'), index=False, header=False)

    test_images = []
    test_classes = []

    for i in range(len(X_test['Frames'])):
        test_images.append('Movies_Frames/' +
                           X_test['Frames'].iloc[i].split('?')[1] +
                           '/' +
                           X_test['Frames'].iloc[i].split('?')[0] +
                           '/' +
                           X_test['Frames'].iloc[i].split('?')[2]
                           )
        test_classes.append(X_test['Frames'].iloc[i].split('?')[0])

    test_df = pd.DataFrame()
    test_df['Frames'] = test_images
    test_df['Movie'] = test_classes
    if os.path.exists(os.path.join(CSV_FILE_PATH, 'Test_Movies.csv')):
        os.remove(os.path.join(CSV_FILE_PATH, 'Test_Movies.csv'))
    test_df.to_csv(os.path.join(CSV_FILE_PATH, 'Test_Movies.csv'), index=False, header=False)

# 3. rulez asta ca sa imi creez csv-urile de train si test
# split_data_to_train_test()
