from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.python.framework import ops
import tensorflow.keras.layers as tfl
import tensorflow as tf
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
import scipy
from PIL import Image
import os


def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y


DATASET_IMAGES = r'dog_images/'
dog_breeds = os.listdir(DATASET_IMAGES)
dog_breeds.sort()

X = []
y = []
for i, dog in enumerate(dog_breeds):
    images = os.listdir(DATASET_IMAGES + dog)
    for image in images:
        img = Image.open(DATASET_IMAGES + dog + '/' + image)
        img = np.array(img.resize((64, 64)))  # resizing the image to 64x64
        X.append(img)
        y.append(i)

X = np.array(X)
y = np.array(y).reshape(-1, 1)  # reshape y to (number_examples, 1)
# print("Shape of X:", X.shape)
# print("Shape of y:", y.shape)
X_train_orig,  X_test_orig, y_train_orig, y_test_orig = train_test_split(
    X, y, test_size=0.2)
# print(X_train_orig.shape)
# print(y_train_orig.shape)
# print(X_test_orig.shape)
# print(y_test_orig.shape)

# index = 124
# plt.imshow(X_train_orig[index])  # display sample training image
# plt.show()

X_train = X_train_orig/255.
X_test = X_test_orig/255.

# create an instance of OneHotEncoder
encoder = OneHotEncoder(sparse=False)

# fit and transform y_train_orig
y_train = encoder.fit_transform(y_train_orig)

# transform y_test_orig using the same encoder instance
y_test = encoder.transform(y_test_orig)

# print("Shape of y_train:", y_train.shape)
# print("Shape of y_test:", y_test.shape)


def convolutional_model(input_shape):
    """
    Implements the forward propagation for the model:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> DENSE

    Arguments:
    input_img -- input dataset, of shape (input_shape)

    Returns:
    model -- TF Keras model (object containing the information for the entire training process) 
    """

    input_img = tf.keras.Input(shape=input_shape)
    Z1 = tfl.Conv2D(filters=8, kernel_size=(4, 4),
                    strides=(1, 1), padding='same')(input_img)
    A1 = tfl.ReLU()(Z1)
    P1 = tfl.MaxPool2D(pool_size=(8, 8), strides=8, padding='same')(A1)
    Z2 = tfl.Conv2D(filters=16, kernel_size=(
        2, 2), strides=1, padding='same')(P1)
    A2 = tfl.ReLU()(Z2)
    P2 = tfl.MaxPool2D(pool_size=(4, 4), strides=4, padding='same')(A2)
    F = tfl.Flatten()(P2)
    outputs = tfl.Dense(units=10, activation='softmax')(F)
    model = tf.keras.Model(inputs=input_img, outputs=outputs)
    return model


conv_model = convolutional_model((64, 64, 3))
conv_model.compile(optimizer='adam',
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])
conv_model.summary()

train_dataset = tf.data.Dataset.from_tensor_slices(
    (X_train, y_train)).batch(64)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(64)
print(test_dataset)
history = conv_model.fit(train_dataset, epochs=800,
                         validation_data=test_dataset)
