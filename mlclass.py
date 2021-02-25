#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 12:16:48 2021

@author: operator
"""

from sklearn import metrics
from sklearn.model_selection import train_test_split

class ml:
    
    def __init__(self, df, y):
        
        self.df = df
        
        self.y = df[[y]]
        self.x = df[[i for i in df.columns if i != y]]
        
        self.xtrain, self.xval, self.ytrain, self.yval = train_test_split(self.x, self.y, random_state = 100, test_size = .2)
        
    def implement_model(self, name, import_statement, model_init):
        
        exec(import_statement, globals())   
        exec(model_init, globals())
        
        model.fit(self.xtrain, self.ytrain)
        
        # Evaluate
        preds = model.predict(self.xval)

        score = metrics.roc_auc_score(self.yval, preds)
        
        print(f'AUC for {name}: {round(score, 3)}')
        print('Confusion Matrix-')
        print(metrics.confusion_matrix(self.yval, preds))
        