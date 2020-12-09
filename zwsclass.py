#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 13:33:00 2020

@author: specialist
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from textblob import TextBlob
import gensim
import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE
    
# Initialize
stop_words = set(stopwords.words('english'))
words = set(nltk.corpus.words.words())
    
'''
    General Machine Learning
'''

# Build class for model
class zws:
    
    # Initialize
    def __init__(self, fp, cat_cols):
        
        self.fp = fp
        self.df = pd.read_csv(fp)
        self.cat_cols = cat_cols
    
    # Test
    def say_hi(self):
        
        print('Hi! I am a Dataset, here are my features:')
        print(self.df.columns)
        print('My measurements are: ', self.df.shape)
        print('Heres a peak: ')
        print(self.df.head())
        
        return self.df
    
    # Function to convert cats
    def convert_cats(self):
        
        le = LabelEncoder()
        
        for i in cat_cols:
            
            df[i] = le.fit_transform(df[i])
            
    # Function to create viz
    def viz_prod(self, title, x, y):
        
        plt.figure()
        plt.title(title)
        plt.plot(x, y)

'''
    Image Classification
'''

# Build class for image recognition
class images:
    
    # Initialize
    def __init__(self, trainpath, valpath, normalization):
        
        self.trainpath = trainpath
        self.valpath = valpath
        self.normalization = normalization
        
# Function to get CNN
    def get_cnn():
        
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
        
        return model            
        
'''
    NLP
'''

# Build class to handle NLP
class zlp:

    # Initialize   
    def __init__(self, fp, txt_col):

        self.fp = fp
        self.df = pd.read_csv(fp)
        self.txt_cols = txt_cols
        self.txt = df[txt_col]
   
    # Function to clean txt
    def process_txt(x):

        x = x.translate(str.maketrans('', '', string.punctuation))
        x = re.sub('\d+', '', x).lower().split()
        x = [i for i in x if i in words]
        x = [i for i in x if i not in stop_words and len(i) > 3]
    
        return x

    # Function to filter tags
    def filter_pos(x):

        tags = nltk.pos_tag(x)
        x1 = [x[0] for x in tags if x[0] in words and len(x[0]) > 3 and x[1].startswith(('N', 'J', 'V'))]

        if len(x1) > 0:
        
            return ' '.join(x1)
    
        else:
        
            return None

    # Function to get sentiment
    def get_sentiment(x):

        return TextBlob(x).sentiment.polarity

    # Function
    def get_tone(score):
    
        if (score >= 0.1):
        
            label = "positive"
   
        elif (score <= -0.1):
        
            label = "negative"
        
        else:
        
            label = "neutral"
        
        return label
    
    # Build topics using coherence
    def optimize_topics(df):
        
        wordid = corpora.Dictionary(df['pos'].apply(lambda x: x.split()))
        corpus = [wordid.doc2bow(txt.split()) for txt in df['pos']]
        scores = []

        for n in range(1, 15, 1):
    
        # Build optimized model
            model = gensim.models.ldamodel.LdaModel(corpus = corpus,
                                                    id2word = wordid,
                                                    num_topics = n,
                                                    random_state = 100)

            cm = CoherenceModel(model = model, corpus = corpus, coherence = 'u_mass')
            scores.append(cm.get_coherence())
    
        # Visualize to choose optimal
        plt.figure()
        plt.plot([i for i in range(1, 15, 1)], scores)
        plt.xlabel('# Topics')
        plt.ylabel('Coherence')
        plt.title('Coherence Scores for LDA w/ n Topics')
        plt.savefig('coherence.png')
        
        return wordid, corpus
    
    # Function to build topic model
    def build_topics(n, df):
        
        wordid = corpora.Dictionary(df['pos'].apply(lambda x: x.split()))
        corpus = [wordid.doc2bow(txt.split()) for txt in df['pos']]
        
        # Build optimized model
        model = gensim.models.ldamodel.LdaModel(corpus = corpus,
                                                id2word = wordid,
                                                num_topics = n,
                                                random_state = 100)

        # Get dominant topics
        topics = []

        for row in model[corpus]:
    
            topics.append(row)
    
        dom_topics = []

        for doc in topics:
    
            dom_topics.append(sorted(doc, key = lambda x: x[1], reverse = True)[0][0])

        # Assign
        df['topic'] = dom_topics
        print(df['topic'].value_counts())
        
        return df
        
    # Function to get TSNE
    def build_tsne(df):
        
        # t-SNE
        vec = CountVectorizer(stop_words = None)
        
        x = vec.fit_transform(df['pos'])
            
        tsne = TSNE(n_components = 2, random_state = 100).fit_transform(x)

        df['t1'] = tsne[:, 0]
        df['t2'] = tsne[:, 1]

        # Visualize
        plt.figure()
        plt.scatter(df['t1'], df['t2'], df['topic'])
        plt.xlabel('T1')
        plt.ylabel('T2')
        plt.title('tSNE Scores for LDA by Topic')
        plt.savefig('tsne.png')
        
        return df
    