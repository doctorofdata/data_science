#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 13:57:07 2020

@author: operator
"""

# Import libraries
import pandas_datareader.data as pdr
import pandas as pd
import numpy as np
from pylab import mpl, plt
mpl.rcParams['savefig.dpi'] = 500
mpl.rcParams['font.family'] = 'serif'
plt.style.use('seaborn')
mpl.rcParams['figure.figsize'] = [10, 6]
from operator import itemgetter
import itertools
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, RMSprop
import warnings
warnings.filterwarnings('ignore')
import random

# Fetch data
# Gold - GC=F
df = pdr.get_data_yahoo('EURUSD=X', '2010-01-01', '2020-12-23').rename({'Adj Close': 'price'}, axis = 1)
df['returns'] = np.log(df['price'] / df['price'].shift(1))
df['creturns'] = df['returns'].cumsum().apply(np.exp)

# Add signal for market dir
df['dir'] = np.where(df['returns'] > 0, 1, 0)

# Function to set seeds
def set_seeds():
    
    random.seed(100) 
    np.random.seed(100)
    #tf.random.set_seed(100)
    
# Create lags
lags = 10

cols = []

for lag in range(1, lags+1):
    
    col = f'lag_{lag}'
    df[col] = df['returns'].shift(lag)
    cols.append(col)
    
df.dropna(inplace = True)

# Round values
df1 = df.round(4)

# Initialize nn
opt = Adam(learning_rate = .0001)

set_seeds()
m = Sequential()
m.add(Dense(64, activation = 'relu', input_shape = [lags,]))
m.add(Dense(64, activation = 'relu'))
m.add(Dense(1, activation = 'sigmoid'))
m.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = ['accuracy'])
m.fit(df1[cols], df1['dir'], epochs = 50, validation_split = .2)
m.evaluate(df1[cols], df1['dir'])

# Predict
preds = np.where(m.predict(df1[cols]) > .5, 1, 0)
df1['pred'] = np.where(preds > 0, 1, -1)
df1['strategy'] = (df1['pred'] * df1['returns'])

# Evaluate
df1[['returns', 'strategy']].cumsum().apply(np.exp).plot(figsize = (10, 6), title = 'Relative Return of ML Strategy Compared to Stock Performance')
