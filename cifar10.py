#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 07:19:45 2020

@author: specialist
"""

# Import libraries
import numpy
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.constraints import maxnorm
from keras.utils import np_utils
from keras.datasets import cifar10
seed = 100

# Load data
(xtrain, ytrain), (xval, yval) = cifar10.load_data()

# Normalize images
xtrain = xtrain.astype('float32')
xval = xval.astype('float32')
xtrain = xtrain / 255.0
xval = xval / 255.0

# One hot encode target
ytrain = np_utils.to_categorical(ytrain)
yval = np_utils.to_categorical(yval)
class_num = yval.shape[1]

# Initialize model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape = xtrain.shape[1:], padding = 'same'))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3), input_shape = (3, 32, 32), activation = 'relu', padding = 'same'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), padding = 'same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), padding = 'same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.2))
model.add(BatchNormalization())  
model.add(Conv2D(128, (3, 3), padding = 'same'))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(256, kernel_constraint = maxnorm(3)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(128, kernel_constraint = maxnorm(3)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(class_num))
model.add(Activation('softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

# Fit model
model.fit(xtrain, ytrain, validation_data = (xval, yval), epochs = 25, batch_size = 64)

# Evaluate
scores = model.evaluate(xval, yval, verbose = 0)
print("Accuracy: %.2f%%" % (scores[1] * 100))
