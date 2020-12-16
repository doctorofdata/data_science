#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 09:07:17 2020

@author: operator
"""

# Import libraries
import pandas as pd
import os
os.chdir('/Users/operator/Documents/code')
import zws
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder
import string
import re
from gensim import corpora
from collections import defaultdict
from gensim.models.phrases import Phrases, Phraser

# Read data
df = pd.read_csv('/Users/operator/Documents/data/netflix_titles.csv')
df1 = df.dropna(subset = ['director', 'cast'])

# Combine casts
ensembles = df1.apply(lambda row: row['director'] + ' ' + row['cast'], axis = 1)
e1 = [e.split(', ') for e in ensembles]

# Calculate frequencies
freqs = defaultdict(int)

for cast in e1:
    for i in cast:
        freqs[i] += 1
        
# Top 10
sorted(freqs, key = freqs.get, reverse = True)[:10]

'''
    Market Basket Analysis- Apriori Association Rules
    
        Confidence = P(B|A)
        Support = P(AB)
        Lift = P(B|A) / P(B)
'''

# Encode casts to product baskets
te = TransactionEncoder()
baskets = te.fit(e1).transform(e1)
b1 = pd.DataFrame(baskets, columns = te.columns_)

# Calculate frequency %s
fxs = apriori(b1, min_support = .001, use_colnames = True)

# Display top 10
fxs.sort_values('support', ascending = False)[0:10]

# Calculate rules - lift = (P|A&B) / (P(A) * P(B))
rules = association_rules(fxs, metric = 'support', min_threshold = 0).sort_values('support', ascending = False).reset_index(drop = True)

'''
    Movie Recommendations
'''

txt = df.apply(lambda row: ', '.join([row['title'], row['description']]), axis = 1)
txt1 = [x.translate(str.maketrans('', '', string.punctuation)) for x in txt]
txt2 = [re.sub('\d+', '', x).lower().split() for x in txt1]
dictionary = corpora.Dictionary(txt2)

'''
    Alt
'''

e2 = Phrases(ensembles, min_count = 1, progress_per = 10000)
bigrams = Phraser(e2)