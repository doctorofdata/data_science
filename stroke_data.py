#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 10:05:35 2021

@author: operator
"""

# Import 
import pandas as pd

import os
os.chdir('/Users/operator/Documents')
from modeler import *
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Get
df = pd.read_csv('/Users/operator/Documents/stroke_data.csv').drop('id', axis = 1).dropna()

# Encode categorical columns and split for training
cats = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
data = data_refinery(df, cats, 'stroke')

# Oversampling with SMOTE
sm = SMOTE(random_state = 100)
xtrain, ytrain = sm.fit_resample(data.xtrain, data.ytrain.ravel())

models = []

models.append(['LogisticRegression', LogisticRegression(random_state = 100)])
models.append(['SVM', SVC(random_state = 100)])
models.append(['KNeighbors', KNeighborsClassifier()])
models.append(['GaussianNB', GaussianNB()])
models.append(['BernoulliNB', BernoulliNB()])
models.append(['DecisionTree', DecisionTreeClassifier(random_state = 100)])
models.append(['RandomForest', RandomForestClassifier(random_state = 100)])
models.append(['XGBoost', XGBClassifier(random_state = 100, eval_metric = 'error')])

pipeline = modeling_pipeline(models)