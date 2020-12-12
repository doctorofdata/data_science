#!/usr/local/Caskroom/miniconda/base/envs/py37
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 06:58:07 2020

@author: operator
"""

# Import libraries
import os
import pandas as pd
import zipfile
import numpy as np
from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from glob import glob
import matplotlib.pyplot as plt

# Define working directory
wd = '/Users/operator/Documents/data/chest_xray/'
os.chdir(wd)

# Load transfer model
vgg = VGG16(input_shape = [224, 224] + [3], weights = 'imagenet', include_top = False)

for layer in vgg.layers:
    
    layer.trainable = False
    
# Define the data
folders = glob('/Users/operator/Documents/data/chest_xray/train/*')
x = Flatten()(vgg.output)
pred = Dense(len(folders), activation = 'softmax')(x)
m = Model(inputs = vgg.input, outputs = pred)
m.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

# Initialize model componewnts
trainer = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
tester = ImageDataGenerator(rescale = 1./255)

# Find data
train = trainer.flow_from_directory('/Users/operator/Documents/data/chest_xray/train/',
                                    target_size = (224, 224),
                                    batch_size = 32,
                                    class_mode = 'categorical')

val = trainer.flow_from_directory('/Users/operator/Documents/data/chest_xray/val/',
                                  target_size = (224, 224),
                                  batch_size = 32,
                                  class_mode = 'categorical')

test = trainer.flow_from_directory('/Users/operator/Documents/data/chest_xray/val/',
                                   target_size = (224, 224),
                                   batch_size = 32,
                                   class_mode = 'categorical')

r = m.fit_generator(train, validation_data = val, epochs = 5, steps_per_epoch = len(train), validation_steps= len(val))

