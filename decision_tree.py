# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 18:10:58 2020

@author: sharm
"""

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import tree
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
print(df['target'])
X_train, X_test, Y_train, Y_test = train_test_split(df[data.feature_names], df['target'], random_state=0)

from sklearn.tree import DecisionTreeClassifier 
clf = DecisionTreeClassifier(max_depth= 2 , random_state=0)
tree.plot_tree(clf.fit(X_train,Y_train))
clf.predict(X_test.iloc[0].values.reshape(1,-1))
clf.predict(X_test[0:10])
score = clf.score(X_test,Y_test)
print(score)
max_depth_range = list(range(1,6))
accuracy = []
for depth in max_depth_range:
    clf = DecisionTreeClassifier(max_depth = depth,random_state = 0)
    tree.plot_tree(clf.fit(X_train,Y_train))
    score = clf.score(X_test,Y_test)
    accuracy.append(score)
    
importances = pd.DataFrame({'feature':X_train.columns,'importance':np.round(clf.feature_importances_,3)})
importances = importances.sort_values('importance',ascending=False)
print(importances)
