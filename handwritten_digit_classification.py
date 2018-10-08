# -*- coding: utf-8 -*-
"""
@author: Mohammed Yusuf Khan

Handwritten Digit Classification
"""

## Import all the dependencies
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import mnist
from keras.utils import np_utils
import matplotlib.pyplot as plt


## Load the dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

## Print the shape and size of the data
print("Shape of X_train: {}".format(X_train.shape))
print("Shape of X_test: {}".format(X_test.shape))
print("Size of X_train: {}".format(X_train.size))
print("Size of X_test: {}".format(X_test.size))

## Visualize a few digits
plt.figure(figsize=(8,8))
grid_param = 221
for i in range(0,4):
  plt.subplot(grid_param)
  plt.imshow(X_train[i], cmap=plt.get_cmap('gray'))
  plt.title('Label:{}'.format(y_train[i]))
  grid_param += 1

plt.show()

## Data Normalization
to_predict = X_test[0]
num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')


X_train = X_train / 255
X_test = X_test / 255

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

## Building an Artificial Neural Network to classify Handwitten Digits

#Intialize the model
ann = Sequential()

#Input Layer
ann.add(Dense(16, input_dim=num_pixels, activation='relu'))

#Hidden Layer
ann.add(Dense(16,  activation='relu'))

#Output Layer
ann.add(Dense(10, activation='softmax'))

#Compile and Fit
ann.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
ann.fit(X_train, y_train, epochs=10, batch_size=10)

## Prediction
print("Number to Predict")
plt.imshow(to_predict)

print("ANN Prediction:{}".format(ann.predict_classes(X_test)[0]))

## Ends here