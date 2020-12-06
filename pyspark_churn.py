#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 12:15:49 2020

@author: specialist
"""

# Import libraries
import pandas as pd
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext, SparkSession
from pyspark.ml.feature import StringIndexer, VectorIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier
from pyspark.ml import Pipeline
from pyspark.sql.types import IntegerType
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.sql.functions import isnan, when, count, col
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# Initialize spark
sc = SparkContext.getOrCreate(SparkConf().setMaster("local[*]"))
spark = SparkSession.builder.getOrCreate()

# Read data
df = spark.read.options(header = True).csv('/Users/specialist/Documents/Data/BankChurners.csv')
cols = [x.lower() for x in df.columns]
df = df.toDF(*cols)

# Drop
drop_cols = ['clientnum',
             'naive_bayes_classifier_attrition_flag_card_category_contacts_count_12_mon_dependent_count_education_level_months_inactive_12_mon_1',
             'naive_bayes_classifier_attrition_flag_card_category_contacts_count_12_mon_dependent_count_education_level_months_inactive_12_mon_2']

df1 = df.drop(*drop_cols)

# Inspect
df1.count()

# Handle nulls
df1.select([count(when(isnan(c), c)).alias(c) for c in df1.columns]).show()

'''
    PySpark ML prep
'''

# Label columns
cats = ['gender',
        'education_level',
        'marital_status',
        'income_category',
        'card_category']

nums = [i for i in df.columns if i not in cats and i != 'attrition_flag']
nums.remove('clientnum')

for i in nums:
    
    if i.startswith('naive'):
        
        nums.remove(i)
        
    else:
        
        pass

# Convert string to numeric
for col in nums:
    
    df1 = df1.withColumn(col, df1[col].cast(IntegerType()))
        
# Initialize model components
stages = []

for col in cats:
    
    cat_indexer = StringIndexer(inputCol = col, outputCol = col + '_idx', handleInvalid = 'keep')
    stages += [cat_indexer]

labels = StringIndexer(inputCol = 'attrition_flag', outputCol = 'label')

stages += [labels]

assembly_materials = [c + '_idx' for c in cats] + nums

assembler = VectorAssembler(inputCols = assembly_materials, outputCol = "features")

stages += [assembler]

# Assemble model
pipeline = Pipeline(stages = stages)
model = pipeline.fit(df1)
df2 = model.transform(df1)

'''
    PySpark ML
'''

# Initialize scoring mechanism
evaluator = BinaryClassificationEvaluator()

# Split data for training
train, val = df2.randomSplit([0.7, 0.3], 100)

# Model
lr = LogisticRegression(featuresCol = 'features', labelCol = 'label')
lrm = lr.fit(train)
summary = lrm.summary
preds = lrm.transform(val)
print('Logistic Regression AUC: ' + str(evaluator.evaluate(preds, {evaluator.metricName: "areaUnderROC"})))

# Model
rfc = RandomForestClassifier(seed = 100)
rfcm = rfc.fit(train)
preds = rfcm.transform(val)
print('Random Forest AUC: ' + str(evaluator.evaluate(preds, {evaluator.metricName: "areaUnderROC"})))

# Model
gbt = GBTClassifier(seed = 100)
gbtm = gbt.fit(train)
preds = gbtm.transform(val)
print('GBT model AUC: ' + str(evaluator.evaluate(preds, {evaluator.metricName: "areaUnderROC"})))

'''
    Hyper-paramter Optimization
'''

grid = (ParamGridBuilder().addGrid(gbt.maxDepth, [2, 4, 6]).addGrid(gbt.maxBins, [20, 60]).addGrid(gbt.maxIter, [10, 20]).build())
cv = CrossValidator(estimator=gbt, estimatorParamMaps = grid, evaluator = evaluator, numFolds = 5)
cvm = cv.fit(train)
preds = cvm.transform(val)
print('optimized GBT model AUC: ' + str(evaluator.evaluate(preds, {evaluator.metricName: "areaUnderROC"})))
