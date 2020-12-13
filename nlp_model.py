#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 15:52:26 2020

@author: operator
"""

# Import libraries
import ast
import os
import zlp
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
import numpy as np
import time

# Function to read inputs
def read_preferences():
    
    with open(input('Enter the filepath for the configuration details: '), 'r') as f:
    
        params = ast.literal_eval(f.read())
    
    return params
    
# Function to read into df
def fetch_corpus(fp, s):
    
    df = pd.read_csv(fp, sep = s)
    
    return df

# Function to transform data and build LDA
def get_optimal_topics(params):
    
    vec = CountVectorizer(analyzer = 'word',
                          min_df = 100,
                          lowercase = True,
                          max_features = 50000)

    data = vec.fit_transform(df1['pos'])

    # LDA
    lda = LatentDirichletAllocation(max_iter = 5, learning_method = 'online', learning_offset = 50, random_state = 100)
    model = GridSearchCV(lda, param_grid = params)
    model.fit(data)
    
    # Select optima from gridsearch
    best = model.best_estimator_
    print("Ideal Params for Corpus: ", model.best_params_)
    print("Best Log Likelihood Score: ", model.best_score_)
    print("Model Perplexity: ", best.perplexity(data))
    
    # Assign topics
    out = best.transform(data)

    topicnames = ['Topic' + str(i) for i in range(best.n_components)]
    docnames = ['Doc' + str(i) for i in range(data.shape[0])]

    scores = pd.DataFrame(np.round(out, 2), columns = topicnames, index = docnames)

    dom = np.argmax(scores.values, axis = 1)
    
    # Assign words to topics
    keys = pd.DataFrame(best.components_)
    keys.columns = vec.get_feature_names()
    keys.index = topicnames

    # Function to find dominant words by topic
    def show_topics():
    
        keywords = np.array(vec.get_feature_names())
        topic_keywords = []
    
        for topic_weights in best.components_:
        
            top_keyword_locs = (-topic_weights).argsort()[:10]
            topic_keywords.append(keywords.take(top_keyword_locs))
    
        return topic_keywords

    # Apply function 
    topic_keywords = show_topics()

    # Convert df
    df_topic_keywords = pd.DataFrame(topic_keywords)
    df_topic_keywords.columns = ['Word '+str(i) for i in range(df_topic_keywords.shape[1])]
    df_topic_keywords.index = ['Topic '+str(i) for i in range(df_topic_keywords.shape[0])]
    
    return dom, df_topic_keywords
    
# Run program from CLI
if __name__ == "__main__":
    
    print('Welcome to the NLP zone!')
    
    # Load data
    params = read_preferences()
    os.chdir(params['wd'])
    df = fetch_corpus(params['fn'], params['sep'])
    
    # Inspect
    print('Dataset loaded..')
    print('Dataset contains {} observations'.format(df.shape[0]))
    print('Available features- ')
    print(df.columns)

    # Process
    print('Sample text- ')
    print(df[params['txt_col']].iloc[0])
    print('Begin processing..')

    df['txt'] = df[params['txt_col']].apply(zlp.process_txt)
    df['pos'] = df['txt'].apply(zlp.filter_pos)

    print('Processing complete..')
    print('Sample output- ')
    print(df['pos'].iloc[0])

    # Build LDA models
    df1 = df.dropna(subset = ['pos'])
    
    # Define params for gridsearch
    params = {'n_components': [1, 2, 4, 8, 16, 20], 'learning_decay': [.5, .7, .9]}
    
    print('Building optimized LDA..')
    
    start = time.time()
    
    # Execute model
    topics, keywordsdf = get_optimal_topics(params)
    
    end = time.time()
    
    print('Optimization took {} min..\nComplete!'.format(round(end - start) / 60))
    
    # Assign results
    df1['topic'] = topics
    print('Topic Distribution from LDA:')
    print(df1['topic'].value_counts())
    
    # Output
    print('Topical Compositions: ')
    
    for idx, row in keywordsdf.iterrows():
        
        print('Topic {}- {}'.format(idx, ' '.join([row[col] for col in keywordsdf.columns])))
        
    print('Operation NLP Complete!')
    