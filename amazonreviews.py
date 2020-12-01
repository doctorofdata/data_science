#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 12:02:04 2020

@author: operator
"""

# Import libraries
import pandas as pd
import zxs
import matplotlib.pyplot as plt
import gensim
import gensim.corpora as corpora
from operator import itemgetter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.manifold import TSNE

# Read data
data = pd.read_csv('/Users/operator/Documents/data/amazonreviews.tsv', sep = '\t')

# Process text
data['txt'] = data['review'].apply(zxs.process_txt)
data['txt'] = data['review'].apply(zxs.check_english)
data['pos'] = data['txt'].apply(zxs.filter_pos)
    
# Inspect
print('Raw text - {}'.format(data['review'].iloc[0]))
print('\nProcessed - {}'.format(data['txt'].iloc[0]))    
print('\nFiltered - {}'.format(data['pos'].iloc[0]))    

# Build topics using coherence
wordid = corpora.Dictionary(data['pos'])
corpus = [wordid.doc2bow(txt) for txt in data['pos']]
scores = []

for n in range(1, 15, 1):
    
    scores.append(zxs.get_coherence(n, corpus, wordid, data['pos']))
    
# Visualize to choose optimal
plt.figure()
plt.plot([i for i in range(1, 15, 1)], scores)
plt.xlabel('# Topics')
plt.ylabel('Coherence')
plt.title('Coherence Scores for LDA w/ n Topics')

# Build optimized model
model = gensim.models.ldamodel.LdaModel(corpus = corpus,
                                        id2word = wordid,
                                        num_topics = 10,
                                        random_state = 100)

# Get dominant topics
topics = []

for row in model[corpus]:
    
    topics.append(row)
    
dom_topics = []

for doc in topics:
    
    dom_topics.append(sorted(doc, key = lambda x: x[1], reverse = True)[0][0])

# Assign
data['topic'] = dom_topics
data['topic'].value_counts()

# t-SNE
x = TfidfVectorizer().fit_transform(data['pos'].apply(lambda x: ' '.join(x)))
tsne = TSNE(n_components = 2, random_state = 100).fit_transform(x)

data['t1'] = tsne[:, 0]
data['t2'] = tsne[:, 1]

# Visualize
plt.figure()
plt.scatter(data['t1'], data['t2'], data['topic'])
plt.xlabel('T1')
plt.ylabel('T2')
plt.title('tSNE Scores for LDA by Topic')