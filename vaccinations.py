#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 09:50:15 2020

@author: operator
"""

# Import libraries
import pandas as pd
%run '/Users/operator/Documents/code/zxs.py'
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

# Read data
fts = pd.read_csv('/Users/operator/Documents/data/flu_training_features.csv')
labs = pd.read_csv('/Users/operator/Documents/data/flu_training_labels.csv')
xtest = pd.read_csv('/Users/operator/Documents/data/flu_test_features.csv')

# View 
inspect_dat(fts)

# Tag numeric columns
nums = [i for i in fts.columns if fts[i].dtypes != 'object'][1:]
cats = [i for i in fts.columns if i not in nums][1:]
        
# Iterate numeric dta
handle_cols = []

# Function to handle nulls
def null_pcts(df, col):
    
    threshold = len(col) * .05
    missing = df[col].isnull().sum() / len(col)
    
    if missing > threshold:
        
        print('Warning: {} has > 5% missing values..'.format(col))
        handle_cols.append(col)
        
for col in cats:
    
    null_pcts(fts, col)
    
# Investigate missing
for col in handle_cols:
    
    print('Investigating ', col, '..')
    print(fts[col].value_counts())
    
    fts[col] = fts[col].fillna('unknown')
    
# Handle missing values for numeric
for col in nums:

    fts[col] = fts[col].fillna(-1)  

print('Model Targets: ')
print(labs.drop('respondent_id', axis = 1).columns)

# Convert cats
le = LabelEncoder()

for col in cats:
    
    fts[col] = le.fit_transform(fts[col])

x = fts.drop('respondent_id', axis = 1)

'''
    H1N1
'''

y = labs['h1n1_vaccine']

# Model
build_mod(x, y, RandomForestClassifier(random_state = 100))
build_mod(x, y, LogisticRegression())
build_mod(x, y, xgb.XGBClassifier(random_state = 100))

# Feature selection
x = eliminate_fts(RandomForestClassifier(random_state = 100), 15, x, y)

# Model
build_mod(x, y, RandomForestClassifier(random_state = 100))
build_mod(x, y, LogisticRegression())
build_mod(x, y, xgb.XGBClassifier(random_state = 100))

'''
    Seasonal
'''

y = labs['seasonal_vaccine']

# Model
build_mod(x, y, RandomForestClassifier(random_state = 100))
build_mod(x, y, LogisticRegression())
build_mod(x, y, xgb.XGBClassifier(random_state = 100))