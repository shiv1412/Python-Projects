# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 18:09:26 2020

@author: sharm
"""

import statsmodels.api as sm
from sklearn import datasets
data = datasets.load_boston()
print(data)
import numpy as np
import pandas as pd
df = pd.DataFrame(data.data,columns = data.feature_names)
target = pd.DataFrame(data.target,columns=["MEDV"])

X = df["RM"]
X= sm.add_constant(X)
y = target["MEDV"]

model = sm.OLS(y,X).fit()
print(model)
predictions = model.predict(X)
print(predictions)
print("shivani")

print(model.summary())


#Model 2

X = df[["RM","LSTAT"]]
y = target["MEDV"]
model2= sm.OLS(y,X).fit()
prediction2 = model2.predict(X)
print(model2.summary())


#model using sklearn
from sklearn import linear_model
from sklearn import datasets
data = datasets.load_boston()
df = pd.DataFrame(data.data,columns=data.feature_names)
target = pd.DataFrame(data.target,columns=["MEDV"])

X = df
y = target["MEDV"]
lm1 = linear_model.LinearRegression()
model3 = lm1.fit(X,y)
print(model3)

prediction3 = lm1.predict(X)
print(prediction3)
print(lm1.score(X,y))
print(lm1.coef_)
print(lm1.intercept_)
