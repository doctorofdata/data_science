#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 12:03:32 2021

@author: operator
"""

# Import 
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn import metrics
from sklearn.model_selection import train_test_split
import os
os.chdir('/Users/operator/Documents/')
from mlclass import *

# Get
df = pd.read_csv('/Users/operator/Downloads/stroke-data.csv')

# Correlation
corrs = df.corr()

# BMI - Age
df['bmi'] = df.groupby('age')['bmi'].apply(lambda x: x.fillna(x.mean()))

# Tag categorical
cats = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']

le = LabelEncoder()

for i in cats:
    
    df[i] = le.fit_transform(df[i])

# Split
xtrain, xval, ytrain, yval = train_test_split(df[[i for i in df.columns if i not in ['stroke', 'id']]], df[['stroke']], test_size = .2, random_state = 100)    

# Model
clf = xgb.XGBClassifier(random_state = 100)
clf.fit(xtrain, ytrain)

# Evaluate
preds = clf.predict(xval)

score = metrics.roc_auc_score(yval, preds)

# Implement
df.drop('id', axis = 1, inplace = True)

modeling = ml(df, 'stroke')

modeling.implement_model('RFC', 'from sklearn.ensemble import RandomForestClassifier', 'model = RandomForestClassifier(random_state = 100)')
