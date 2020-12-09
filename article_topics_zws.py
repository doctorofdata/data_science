#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 11:49:18 2020

@author: specialist
"""

# Import libraries
import os
os.chdir('/Users/specialist/Documents/code/')
from zwsclass import zws, zlp

# Load the data
df = zws('/Users/specialist/Documents/data/articles_data.csv', None)
df = df.say_hi()
df1 = df.dropna(subset = ['content'])

# Handle nulls
df1['author'] = df1['author'].fillna('unknown')
df1['title'] = df1['title'].fillna('no title')
df1['description'] = df1['description'].fillna(' ')

# Combine txt cols
df1['txt'] = [x + ' ' + y + ' ' + z + ' ' + a for x, y, z, a in zip(df1['author'], df1['title'], df1['description'], df1['content'])]

# Process the text
df1['txt'] = df1['txt'].apply(zlp.process_txt)
df1['pos'] = df1['txt'].apply(zlp.filter_pos)
df1['sentiment'] = df1['pos'].apply(lambda x: ' '.join(x)).apply(zlp.get_sentiment)
df1['tone'] = df1['sentiment'].apply(zlp.get_tone)

# Find optimal K value
zlp.optimize_topics(df1)

# Build optimal model
df2 = zlp.build_topics(4, df1)

# TSNE
df3 = zlp.build_tsne(df2)
