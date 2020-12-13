#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 07:39:16 2020

@author: operator
"""

# Import libraries
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.constraints import maxnorm
from sklearn.metrics import classification_report,confusion_matrix
import tensorflow as tf
import cv2
import os
import numpy as np
from tqdm import tqdm
from keras.utils import np_utils

# Read data
labels = ['PNEUMONIA', 'NORMAL']

# Function to load images
def get_img_data(data_dir):
    
    data = [] 
    
    # Iterate
    for label in labels: 
        
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        
        print('Loading images for {} class..'.format(label))
        
        for img in tqdm(os.listdir(path)):
            
            try:
                
                # Load
                img1 = cv2.imread(os.path.join(path, img))[...,::-1] 
                img2 = cv2.resize(img1, (224, 224)) 
                data.append([img2, class_num])
                
            except Exception as e:
                
                print(e)
                
    return np.array(data)

# Apply function to load data
train = get_img_data('/Users/operator/Documents/data/chest_xray/train')
val = get_img_data('/Users/operator/Documents/data/chest_xray/val')
test = get_img_data('/Users/operator/Documents/data/chest_xray/test')

# Function to prep training set
def prep_img_set(img_set):
    
    x, y = [], []
    
    for feature, label in img_set:
  
        x.append(feature)
        y.append(label)

    # Normalize the data
    x = np.array(x) / 255

    x.reshape(-1, 224, 224, 1)
    y = np.array(y)
    
    return x, y

# Apply function
xtrain, ytrain = prep_img_set(train)
xval, yval = prep_img_set(val)
xtest, ytest = prep_img_set(test)

# Data augmentation
datagen = ImageDataGenerator(featurewise_center = False,                       # set input mean to 0 over the dataset
                             samplewise_center = False,                        # set each sample mean to 0
                             featurewise_std_normalization = False,            # divide inputs by std of the dataset
                             samplewise_std_normalization = False,             # divide each input by its std
                             zca_whitening = False,                            # apply ZCA whitening
                             rotation_range = 30,                              # randomly rotate images in the range (degrees, 0 to 180)
                             zoom_range = 0.2,                                 # Randomly zoom image 
                             width_shift_range = 0.1,                          # randomly shift images horizontally (fraction of total width)
                             height_shift_range = 0.1,                         # randomly shift images vertically (fraction of total height)
                             horizontal_flip = True,                           # randomly flip images
                             vertical_flip = False)                            # randomly flip images


datagen.fit(xtrain)

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
model.fit(xtrain, ytrain, validation_data = (xval, yval), epochs = 5, batch_size = 64)

# Evaluate
scores = model.evaluate(xval, yval, verbose = 0)
print("Accuracy: %.2f%%" % (scores[1] * 100))