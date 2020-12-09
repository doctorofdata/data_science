#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 09:01:12 2020

@author: specialist
"""

# Import libraries
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from textblob import TextBlob
import gensim
import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.manifold import TSNE

stop_words = set(stopwords.words('english'))
words = set(nltk.corpus.words.words())

# Read data
df = pd.read_csv('/Users/specialist/Documents/data/articles_data.csv').drop('Unnamed: 0', axis = 1)
df1 = df.dropna(subset = ['content'])

# Handle nulls
df1['author'] = df1['author'].fillna('unknown')
df1['title'] = df1['title'].fillna('no title')
df1['description'] = df1['description'].fillna(' ')

# Combine txt cols
df1['txt'] = [x + ' ' + y + ' ' + z + ' ' + a for x, y, z, a in zip(df1['author'], df1['title'], df1['description'], df1['content'])]

# Function to process txt
def process_txt(x):
    
    x = x.translate(str.maketrans('', '', string.punctuation))
    x = re.sub('\d+', '', x).lower().split()
    x = [i for i in x if i in words]
    x = [i for i in x if i not in stop_words and len(i) > 3]
    
    return x

def filter_pos(x):
    
    tags = nltk.pos_tag(x)
    x1 = [x[0] for x in tags if x[0] in words and len(x[0]) > 3 and x[1].startswith(('N', 'J', 'V'))]

    if len(x1) > 0:
        
        return ' '.join(x1)
    
    else:
        
        return None

def get_sentiment(x):
    
    return TextBlob(x).sentiment.polarity

def get_tone(score):
    
    if (score >= 0.1):
        
        label = "positive"
   
    elif (score <= -0.1):
        
        label = "negative"
        
    else:
        
        label = "neutral"
        
    return label

# Apply function
df1['txt'] = df1['txt'].apply(process_txt)
df1['pos'] = df1['txt'].apply(filter_pos)
df1['sentiment'] = df1['pos'].apply(lambda x: ' '.join(x)).apply(get_sentiment)
df1['tone'] = df1['sentiment'].apply(get_tone)

# Build topics using coherence
wordid = corpora.Dictionary(df1['pos'].apply(lambda x: x.split()))
corpus = [wordid.doc2bow(txt.split()) for txt in df1['pos']]
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

# Build optimized model
model = gensim.models.ldamodel.LdaModel(corpus = corpus,
                                        id2word = wordid,
                                        num_topics = 3,
                                        random_state = 100)

# Get dominant topics
topics = []

for row in model[corpus]:
    
    topics.append(row)
    
dom_topics = []

for doc in topics:
    
    dom_topics.append(sorted(doc, key = lambda x: x[1], reverse = True)[0][0])

# Assign
df1['topic'] = dom_topics
df1['topic'].value_counts()

# t-SNE
vec = CountVectorizer()
out = vec.fit_transform(df1['pos'])
tsne = TSNE(n_components = 2, random_state = 100).fit_transform(out)

df1['t1'] = tsne[:, 0]
df1['t2'] = tsne[:, 1]

# Visualize
plt.figure()
plt.scatter(df1['t1'], df1['t2'], df1['topic'])
plt.xlabel('T1')
plt.ylabel('T2')
plt.title('tSNE Scores for Corpus by Topic')