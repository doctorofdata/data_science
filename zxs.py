#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 08:16:06 2020

@author: operator
"""

# Import libraries
import matplotlib.pyplot as plt

# Function to inspect a df
def inspect_dat(df):
    
    print('Available Features- ')
    print(df.columns)
    print()
    print('Shape of dataset - {}'.format(df.shape))
    print('Missing Values: ')
    print(df.isnull().sum())
    
# Function to convert categories
def cat_converter(df, cats):
    
    # Import the required libraries
    from sklearn.preprocessing import LabelEncoder
    
    # Initialize 
    le = LabelEncoder()

    # Convert cats to nums
    for x in cats:
    
        print('Columns- ', x)
        print(df[x].value_counts())
        print('Converting levels..')
        df[x] = le.fit_transform(df[x])
        
    return df

# Function to assess model performance
def evaluate_mod(yval, preds):

    from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, roc_curve, auc
    
    print('Scoring Metrics- ')
    print(classification_report(yval, preds))
    
    fpr, tpr, _ = roc_curve(yval, preds)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color = 'darkorange', label = 'ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color = 'navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc = "lower right")
    plt.show()
    
    print('AUC for model - {}'.format(round(roc_auc, 3)))
    
# Function to conduct RFE
def eliminate_fts(mod, num_vars, x, y):
    
    from sklearn.feature_selection import RFE
    
    # Initialize
    rfe = RFE(mod, num_vars)
    
    # Transform
    x = rfe.fit_transform(x, y)
    
    return x

# Function to determine optimal clustering
def find_k(df):

    from sklearn.cluster import KMeans

    ssd = []
    k = range(1, 15)
    
    for n in k:
        
        m = KMeans(n_clusters = n, random_state = 100).fit(df)
        ssd.append(m.inertia_)

    plt.plot(k, ssd, 'bx-')
    plt.xlabel('# Clusters')
    plt.ylabel('SSD')
    plt.title('Choosing Optimal K Value') 
    
# Function to process text
def process_txt(x):
    
    import re
    import string
    
    x = x.translate(str.maketrans('', '', string.punctuation))
    x = ''.join([i for i in x if not i.isdigit()]).strip()
    
    return x

# Function to filter for english language
def check_english(x):
    
    from nltk.corpus import words
    from nltk.corpus import stopwords
    
    english = words.words()
    stop_words = set(stopwords.words())
    
    wrds = [i for i in x.split(' ') if i in english]
    wrds = [i for i in wrds if i not in stop_words and len(i) > 3]
    
    return wrds
    
# Function to iterate topics to find optima in pyspark
def get_optimal_topics(ds):
    
    from pyspark.ml.clustering import LDA

    for n in range(2, 15, 1):
        
        print('Building model with {} topics..'.format(n))
        
        lda = LDA(featuresCol = 'fts', k = n)
        m = lda.fit(ds)

        print('Log Perplexity = ', m.logPerplexity(ds))    
    
# Function to build model
def build_mod(x, y, mod):
    
    from sklearn.model_selection import train_test_split
    
    xtrain, xval, ytrain, yval = train_test_split(x, y, test_size = .15, random_state = 100)
    
    mod.fit(xtrain, ytrain)
    
    out = mod.predict(xval)

    evaluate_mod(yval, out)
    