import os
import shutil
import services as MyAppServices

from fastapi import FastAPI, HTTPException, File, UploadFile
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://localhost:3000'],
    allow_credentials=False,
    allow_methods=['*'],
    allow_headers=['*'],
)


@app.post('/predict_sequence/all_models')
async def predict_movie_sequence_using_all_models(video: UploadFile = File(None)):
    if video is None:
        raise HTTPException(status_code=400, detail='The video sequence was not uploaded.')

    accepted_video_extensions = ['.mp4', '.avi', '.mkv']
    file_extension = Path(video.filename).suffix.lower()
    if file_extension not in accepted_video_extensions:
        raise HTTPException(status_code=400, detail='The file extension is not supported.')

    temp_sequence_path = f'Sequence/{video.filename}'
    if os.path.exists(temp_sequence_path):
        os.remove(temp_sequence_path)
    with open(temp_sequence_path, 'wb') as buffer:
        shutil.copyfileobj(video.file, buffer)

    MyAppServices.extract_frames_from_sequence(temp_sequence_path)
    custom_model_3d_cnn_predicted_movie = MyAppServices.custom_model_3d_cnn_prediction()
    custom_model_2d_cnn_predicted_movie = MyAppServices.custom_model_2d_cnn_prediction()
    inceptionv3_predicted_movie = MyAppServices.inceptionv3_prediction()
    resnet50_predicted_movie = MyAppServices.resnet50_prediction()
    vgg16_model_1_predicted_movie = MyAppServices.vgg16_model_1_prediction()
    vgg16_model_2_predicted_movie = MyAppServices.vgg16_model_2_prediction()

    os.remove(temp_sequence_path)

    return {
                'Custom Model 3D-CNN Prediction': custom_model_3d_cnn_predicted_movie,
                'Custom Model 2D-CNN Prediction': custom_model_2d_cnn_predicted_movie,
                'InceptionV3 Prediction': inceptionv3_predicted_movie,
                'ResNet50 Prediction': resnet50_predicted_movie,
                'VGG16 Model 1 Prediction': vgg16_model_1_predicted_movie,
                'VGG16 Model 2 Prediction': vgg16_model_2_predicted_movie
            }


@app.post('/predict_sequence/custom_model_3d_cnn')
def predict_movie_sequence_using_custom_model_3d_cnn(video: UploadFile = File(None)):
    if video is None:
        raise HTTPException(status_code=400, detail='The video sequence was not uploaded.')

    accepted_video_extensions = ['.mp4', '.avi', '.mkv']
    file_extension = Path(video.filename).suffix.lower()
    if file_extension not in accepted_video_extensions:
        raise HTTPException(status_code=400, detail='The file extension is not supported.')

    temp_sequence_path = f'Sequence/{video.filename}'
    if os.path.exists(temp_sequence_path):
        os.remove(temp_sequence_path)
    with open(temp_sequence_path, 'wb') as buffer:
        shutil.copyfileobj(video.file, buffer)

    MyAppServices.extract_frames_from_sequence(temp_sequence_path)
    predicted_movie = MyAppServices.custom_model_3d_cnn_prediction()
    os.remove(temp_sequence_path)

    return {'Custom Model 3D-CNN Prediction': predicted_movie}


@app.post('/predict_sequence/custom_model_2d_cnn')
def predict_movie_sequence_using_custom_model_2d_cnn(video: UploadFile = File(None)):
    if video is None:
        raise HTTPException(status_code=400, detail='The video sequence was not uploaded.')

    accepted_video_extensions = ['.mp4', '.avi', '.mkv']
    file_extension = Path(video.filename).suffix.lower()
    if file_extension not in accepted_video_extensions:
        raise HTTPException(status_code=400, detail='The file extension is not supported.')

    temp_sequence_path = f'Sequence/{video.filename}'
    if os.path.exists(temp_sequence_path):
        os.remove(temp_sequence_path)
    with open(temp_sequence_path, 'wb') as buffer:
        shutil.copyfileobj(video.file, buffer)

    MyAppServices.extract_frames_from_sequence(temp_sequence_path)
    predicted_movie = MyAppServices.custom_model_2d_cnn_prediction()
    os.remove(temp_sequence_path)

    return {'Custom Model 2D-CNN Prediction': predicted_movie}


@app.post('/predict_sequence/inceptionv3')
def predict_movie_sequence_using_inceptionv3(video: UploadFile = File(None)):
    if video is None:
        raise HTTPException(status_code=400, detail='The video sequence was not uploaded.')

    accepted_video_extensions = ['.mp4', '.avi', '.mkv']
    file_extension = Path(video.filename).suffix.lower()
    if file_extension not in accepted_video_extensions:
        raise HTTPException(status_code=400, detail='The file extension is not supported.')

    temp_sequence_path = f'Sequence/{video.filename}'
    if os.path.exists(temp_sequence_path):
        os.remove(temp_sequence_path)
    with open(temp_sequence_path, 'wb') as buffer:
        shutil.copyfileobj(video.file, buffer)

    MyAppServices.extract_frames_from_sequence(temp_sequence_path)
    predicted_movie = MyAppServices.inceptionv3_prediction()
    os.remove(temp_sequence_path)

    return {'InceptionV3 Prediction': predicted_movie}


@app.post('/predict_sequence/resnet50')
def predict_movie_sequence_using_resnet50(video: UploadFile = File(None)):
    if video is None:
        raise HTTPException(status_code=400, detail='The video sequence was not uploaded.')

    accepted_video_extensions = ['.mp4', '.avi', '.mkv']
    file_extension = Path(video.filename).suffix.lower()
    if file_extension not in accepted_video_extensions:
        raise HTTPException(status_code=400, detail='The file extension is not supported.')

    temp_sequence_path = f'Sequence/{video.filename}'
    if os.path.exists(temp_sequence_path):
        os.remove(temp_sequence_path)
    with open(temp_sequence_path, 'wb') as buffer:
        shutil.copyfileobj(video.file, buffer)

    MyAppServices.extract_frames_from_sequence(temp_sequence_path)
    predicted_movie = MyAppServices.resnet50_prediction()
    os.remove(temp_sequence_path)

    return {'ResNet50 Prediction': predicted_movie}


@app.post('/predict_sequence/vgg16_model_1')
def predict_movie_sequence_using_vgg16_model_1(video: UploadFile = File(None)):
    if video is None:
        raise HTTPException(status_code=400, detail='The video sequence was not uploaded.')

    accepted_video_extensions = ['.mp4', '.avi', '.mkv']
    file_extension = Path(video.filename).suffix.lower()
    if file_extension not in accepted_video_extensions:
        raise HTTPException(status_code=400, detail='The file extension is not supported.')

    temp_sequence_path = f'Sequence/{video.filename}'
    if os.path.exists(temp_sequence_path):
        os.remove(temp_sequence_path)
    with open(temp_sequence_path, 'wb') as buffer:
        shutil.copyfileobj(video.file, buffer)

    MyAppServices.extract_frames_from_sequence(temp_sequence_path)
    predicted_movie = MyAppServices.vgg16_model_1_prediction()
    os.remove(temp_sequence_path)

    return {'VGG16 Model 1 Prediction': predicted_movie}


@app.post('/predict_sequence/vgg16_model_2')
def predict_movie_sequence_using_vgg16_model_2(video: UploadFile = File(None)):
    if video is None:
        raise HTTPException(status_code=400, detail='The video sequence was not uploaded.')

    accepted_video_extensions = ['.mp4', '.avi', '.mkv']
    file_extension = Path(video.filename).suffix.lower()
    if file_extension not in accepted_video_extensions:
        raise HTTPException(status_code=400, detail='The file extension is not supported.')

    temp_sequence_path = f'Sequence/{video.filename}'
    if os.path.exists(temp_sequence_path):
        os.remove(temp_sequence_path)
    with open(temp_sequence_path, 'wb') as buffer:
        shutil.copyfileobj(video.file, buffer)

    MyAppServices.extract_frames_from_sequence(temp_sequence_path)
    predicted_movie = MyAppServices.vgg16_model_2_prediction()
    os.remove(temp_sequence_path)

    return {'VGG16 Model 2 Prediction': predicted_movie}


@app.get('/')
def home():
    return {'message': 'Welcome to the Movie Sequence Prediction API.'}
