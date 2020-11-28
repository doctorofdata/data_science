#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 08:02:44 2020

@author: operator
"""

# Import libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

%run '/Users/operator/Documents/code/zxs.py'

# Read data
df = pd.read_csv('/Users/operator/Documents/data/bank_churn.csv')

# Apply function to get data summary
inspect_dat(df)

# Tag the categorical columns
df.columns = [x.lower() for x in df.columns]

cats = ['gender',
        'education_level',
        'marital_status',
        'income_category',
        'card_category']

# Convert the target
df['attrition_flag'].value_counts()
df['churn'] = df['attrition_flag'].apply(lambda x: 1 if x == 'Attrited Customer' else 0)
df['churn'].value_counts()

# Convert cats to nums
df1 = cat_converter(df, cats)
df1.drop(['clientnum', 'attrition_flag'], axis = 1, inplace = True)

# Variable correlation
sns.heatmap(df1.corr())

# Prevent info leak
df2 = df1[[i for i in df1.columns if not i.startswith('naive')]]

# Split the variables 
y = df2['churn']
x = df2[[i for i in df2.columns if i != 'churn']]

xtrain, xval, ytrain, yval = train_test_split(x, y, test_size = .3, random_state = 100)

# Baseline model
rfc = RandomForestClassifier(random_state = 100)
rfc.fit(xtrain, ytrain)
preds = rfc.predict(xval) 

# Evaluate
evaluate_mod(yval, preds)

# Recursive feature elimination
x1 = eliminate_fts(rfc, round(df.shape[1] / 2, 0), x, y)

# Refit model
xtrain, xval, ytrain, yval = train_test_split(x1, y, test_size = .3, random_state = 100)

rfc.fit(xtrain, ytrain)
preds = rfc.predict(xval) 

# Evaluate
print('Scoring truncated model..')
evaluate_mod(yval, preds)

# Cluster
find_k(x1)

# Build optimal clustering
m = KMeans(n_clusters = 4, random_state = 100)
m.fit(x1)

x2 = pd.DataFrame(x1)
x2['cluster'] = m.labels_

# Split the data
xtrain, xval, ytrain, yval = train_test_split(x2, y, test_size = .3, random_state = 100)

rfc.fit(xtrain, ytrain)
preds = rfc.predict(xval) 

# Evaluate
print('Scoring truncated model..')
evaluate_mod(yval, preds)

# Investigate
importance = rfc.feature_importances_

plt.title('Feature Importance for RF Model w/ Clusters')
plt.bar([x for x in range(len(importance))], importance)
plt.show()
