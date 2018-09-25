import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
import pickle
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


def main():

    X, y, no_classes = load_data(True)

    # conv_net(X, y, no_classes)

    # SVM implementation

    X = X.reshape(*X.shape[:1], -1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.95,
                                                        random_state=42)
    svm = SVC(C=5)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    print(accuracy_score(y_test, y_pred))


def load_data(load=False):
    """ Loads x and y and resizes all the images. If load is True, the data in
    the pickle file is loaded, saving calculation time"""

    if load==True:
        with open('pickle/imData.pkl', 'rb') as f:
            features, labels, no_classes= pickle.load(f)

    elif load == False:
        labels = []
        minWidth, minHeight = image_width(True)
        features = []
        no_classes = 0

        # Loop through all folders and files, resampling so they all have the same
        # size.
        for i, folder_name in enumerate(os.listdir("Data")):
            no_classes += 1
            for file_name in os.listdir("Data/" + folder_name):
                img = Image.open("Data/" + folder_name + "/" + file_name)

                # Resize the image based on the smallest image found. Also
                # antialiase.
                img = img.resize((minWidth, minHeight), Image.ANTIALIAS)
                img_array = np.array(img)

                # Some of the images have alpha transparency or are only B&W.
                # For the time being, I simply remove these
                if img_array.shape == (minHeight, minWidth, 3):
                    features.append(img_array)
                    labels.append(i)

        features = np.array(features)
        labels = np.array(labels)
        with open('pickle/imData.pkl', 'wb') as f:
            pickle.dump([features, labels, no_classes], f)

    return features, labels, no_classes


def image_width(load=False):
    """ Finds the smallest image size in the directory and returns the
    dimensions. Also gives the option to pickle to save time """

    if load==False:
        minWidth = 500
        minHeight = 500
        for folder_name in os.listdir("Data"):
            for file_name in os.listdir("Data/" + folder_name):
                img = Image.open("Data/" + folder_name + "/" + file_name)
                img_array = np.array(img)
                if img_array.shape[0] < minWidth:
                    minWidth = img_array.shape[0]
                if img_array.shape[1] < minHeight:
                    minHeight = img_array.shape[1]
        with open('pickle/imsize.pkl', 'wb') as f:
            pickle.dump([minWidth, minHeight], f)

    if load==True:
        with open('pickle/imsize.pkl', 'rb') as f:
            minWidth, minHeight = pickle.load(f)

    return minWidth, minHeight


def conv_net(X, y, num_classes):
    """ Fit a CNN to the data - this is likely to massively overfit because
    there isn't enough data """

    # Converts the labels to a matrix of categorical values
    labels = keras.utils.to_categorical(labels, no_classes)

    model = Sequential()
    model.add(Conv2D(20, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=(134, 98, 3)))
    model.add(Dropout(0.5))
    model.add(Conv2D(20, kernel_size=(3, 3), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(no_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer='adam',
                  metrics=['accuracy'])
    model.fit(X, y,
              batch_size=128,
              epochs=5,
              validation_split = 0.3)

if __name__=="__main__":
    main()
