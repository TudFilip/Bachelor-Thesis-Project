# Bachelor-Thesis-Project

The main goal of the thesis was to create an artificial intelligence system capable of recognizing a user-entered sequence from an unknown movie to identify the movie title. The AI system is based on 6 models of convolutional neural networks, 4 of them being made using pretrained models (VGG16, InceptionV3 and ResNet50), and the other 2 being made using an own architecture structure.

The structure of the repo branches:
* **ai_system** -> AI system consisting of the 6 convolutional neural network models
* **movies_preprocessin** -> The system used to create the training/test dataset from 10 movies
* **web_application** -> Web application that allows a user to upload a movie clip and receive the title of that clip as a response
* **plots** -> Python scripts used to create result-based plots
