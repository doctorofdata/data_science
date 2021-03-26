#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 10:24:28 2021

@author: operator
"""

# Import 
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, classification_report, roc_curve, plot_roc_curve, auc, precision_recall_curve, plot_precision_recall_curve, average_precision_score

# Class for returning data
class data_refinery:
    
    def __init__(self, df, cats, y):
        
        self.df = df
        
        if cats:
            
            le = LabelEncoder()
            
            self.cats = cats
            
            for i in cats:
                
                self.df[i] = le.fit_transform(self.df[i])
            
        self.x = self.df[[i for i in self.df.columns if i != y]]
        self.y = self.df[y]
        
        self.xtrain, self.xval, self.ytrain, self.yval = train_test_split(self.x, self.y, test_size = .2, random_state = 100)
        
# Class for modeling
class modeling_pipeline:
    
    def __init__(self, models):
        
        self.scores = {}

    def execute_pipeline_flow(self):
                
        for m in range(len(self.models)):
                
            out = []
    
            model = models[m][1]
            model.fit(x_train_res, y_train_res)
            y_pred = model.predict(x_test)
            cm = confusion_matrix(y_test, y_pred)  
            accuracies = cross_val_score(estimator = model, X = x_train_res, y = y_train_res, cv = 10)   
            roc = roc_auc_score(y_test, y_pred)  
            precision = precision_score(y_test, y_pred) 
            recall = recall_score(y_test, y_pred) 
            f1 = f1_score(y_test, y_pred)  
            
            print(models[m][0],':')
            print(cm)
            print('Accuracy Score: ',accuracy_score(y_test, y_pred))
            print('')
            print("K-Fold Validation Mean Accuracy: {:.2f} %".format(accuracies.mean() * 100))
            print('')
            print("Standard Deviation: {:.2f} %".format(accuracies.std() * 100))
            print('')
            print('ROC AUC Score: {:.2f}'.format(roc))
            print('')
            print('Precision: {:.2f}'.format(precision))
            print('')
            print('Recall: {:.2f}'.format(recall))
            print('')
            print('F1: {:.2f}'.format(f1))
            print('-----------------------------------')
            print('')
            
            out.append((accuracy_score(y_test, y_pred)) * 100) 
            out.append(accuracies.mean() * 100)
            out.append(accuracies.std() * 100)
            out.append(roc)
            out.append(precision)
            out.append(recall)
            out.append(f1)
            
            self.scores[models[m][0]] = out