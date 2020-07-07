# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 22:04:32 2020

@author: sharm
"""

import os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#sns.set(style="white")
#sns.set(style="whitegrid", color_codes=True)
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

data = pd.read_csv('https://raw.githubusercontent.com/madmashup/targeted-marketing-predictive-engine/master/banking.csv', sep=',')

data.head(20)
data.describe()
type(data['age'])
data['marital'].unique()
data['poutcome'].unique()
data['month'].unique()
data['job'].unique()
data['education'].unique()

data['education']=np.where(data['education'] =='basic.9y', 'Basic', data['education'])
data['education']=np.where(data['education'] =='basic.6y', 'Basic', data['education'])
data['education']=np.where(data['education'] =='basic.4y', 'Basic', data['education'])

# After grouping the categories the categories look like this
data['education'].unique()
data['y'].value_counts()
sns.countplot(x='y',data=data,palette='hls')
plt.show()
plt.savefig('count_plot')

count_no_sub = len(data[data['y']==0])
count_sub = len(data[data['y']==1])
pct_of_no_sub = count_no_sub/(count_no_sub+count_sub)
print("percentage of no subscription is", pct_of_no_sub*100)
pct_of_sub = count_sub/(count_no_sub+count_sub)
print("percentage of subscription", pct_of_sub*100)

data.groupby('y').mean()

## Correlation
#get correlations of each features in dataset
corrmat = data.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,21))
#plot heat map
g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")
