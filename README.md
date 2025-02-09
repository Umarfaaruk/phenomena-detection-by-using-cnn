
# Pneumonia Detection Using CNN

This repository contains code for detecting pneumonia in chest X-ray images using a Convolutional Neural Network (CNN). The dataset is organized into training and test sets, and the trained model is used to classify images as either pneumonia or normal.

## Introduction
This project involves building a Convolutional Neural Network (CNN) to detect pneumonia from chest X-ray images. The model is trained on a dataset of labeled images and evaluated on a separate test set. The goal is to achieve high accuracy in detecting pneumonia to aid in medical diagnosis.

## Installation
Make sure you have the following packages installed:
```bash
pip install tensorflow keras numpy matplotlib opencv-python
```

## Dataset
The dataset should be organized into the following structure:
```
dataset/
    training_set/
        pneumonia/
        normal/
    test_set/
        pneumonia/
        normal/
```

## Training the Model
Run the following command to train the model:
```bash
python pneumonia_detection.py
```
The script `pneumonia_detection.py` contains the code to preprocess the data, build the CNN model, and train it on the training set. The model's architecture includes multiple convolutional layers, pooling layers, a flattening layer, and fully connected layers.

## Testing the Model
After training, the model can be tested using the test set to evaluate its performance. The test set should contain images that the model has not seen during training. The script will output the accuracy and loss metrics for the test set.

## Evaluation
The performance of the model can be evaluated using various metrics such as accuracy, precision, recall, and F1 score. These metrics provide insights into how well the model is performing in classifying the images correctly.

## Results
After training, the model achieves an accuracy of approximately XX% on the validation set. The performance can be further improved with additional tuning and data augmentation. You can visualize the training and validation accuracy and loss over epochs using matplotlib.

## Contributing
Feel free to contribute to this project by submitting issues or pull requests. Your contributions are welcome!
