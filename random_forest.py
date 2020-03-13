# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 00:17:43 2020

@author: sharm
"""

#random forest algorithm
from sklearn import datasets
import pandas as pd
iris =datasets.load_iris()
print(iris.target_names)
print(iris.features_names)
print(iris.data[0:5])
print(iris.target)
data=pd.DataFrame({
    'sepal length':iris.data[:,0],
    'sepal width':iris.data[:,1],
    'petal length':iris.data[:,2],
    'petal width':iris.data[:,3],
    'species':iris.target
})
data.head()
