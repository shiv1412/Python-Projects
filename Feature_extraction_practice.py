# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 15:27:38 2020

@author: sharm
"""

import pandas as pd
import numpy as np
from numpy import mean
from numpy import std
from pandas import read_csv
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.preprocessing import LabelEncoder,MinMaxScaler,StandardScaler,RobustScaler,QuantileTransformer, KBinsDiscretizer
from sklearn.decomposition import PCA,TruncatedSVD
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/wine.csv'
df = read_csv(url, header = None)
data =df.values
X,y = data[:,:-1],data[:,-1]
print(X.shape,y.shape)

X = X.astype('float')
y = LabelEncoder().fit_transform(y.astype('str'))


transforms = list()
transforms.append(('mms',MinMaxScaler()))
transforms.append(('ss',StandardScaler()))
transforms.append(('rs',RobustScaler()))
transforms.append(('qt',QuantileTransformer(n_quantiles=100,output_distribution='normal')))
transforms.append(('kbd',KBinsDiscretizer(n_bins=10,encode='ordinal',strategy='uniform')))
transforms.append(('pca',PCA(n_components=7)))
transforms.append(('svd',TruncatedSVD(n_components=7)))

fu =FeatureUnion(transforms)
rfe = RFE(estimator=LogisticRegression(solver = 'liblinear'),n_features_to_select=15)
model = LogisticRegression(solver = 'liblinear')
# define the pipeline
steps = list()
steps.append(('fu',fu))
steps.append(('r', rfe))
steps.append(('m',model))
pipeline = Pipeline(steps=steps)

# handling cross validation
cv = RepeatedStratifiedKFold(n_splits=10,n_repeats=3,random_state=1)
#evaluation of model
scores = cross_val_score(pipeline, X,y,scoring= 'accuracy',cv=cv,n_jobs=-1)

print(mean(scores),std(scores))
