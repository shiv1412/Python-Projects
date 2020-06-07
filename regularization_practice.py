# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 09:13:22 2020

@author: sharm
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns 
import warnings 
warnings.filterwarnings('ignore')

url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.data'
crime = pd.read_csv(url, header=None, na_values=['?'])
crime.head()
print(crime[127].describe())

plt.figure(figsize=(20, 6))
sns.heatmap(crime.isnull(),yticklabels=False,cbar=False,cmap='viridis')

crime.drop([0, 1, 2, 3, 4], axis=1, inplace=True)
# removing missing values
crime.dropna(inplace=True)
plt.figure(figsize=(20, 6))
sns.heatmap(crime.isnull(),yticklabels=False,cbar=False,cmap='viridis')

print(crime.shape)

X = crime.drop(127, axis=1)
y = crime[127]

# split into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.3, random_state=1)

# build a linear regression model
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X_train, y_train)
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)
print ("iNTERCEPT : ",linreg.intercept_)
print ("CO-EFFICIENT : ",linreg.coef_)

y_pred = linreg.predict(X_test)
# calculate R^2 value, MAE, MSE, RMSE

from sklearn.metrics import r2_score
from sklearn import metrics
print("R-Square Value",r2_score(y_test,y_pred))
print("\n")
print ("mean_absolute_error :",metrics.mean_absolute_error(y_test, y_pred))
print("\n")
print ("mean_squared_error : ",metrics.mean_squared_error(y_test, y_pred))
print("\n")
print ("root_mean_squared_error : ",np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# alpha=0 is equivalent to linear regression
from sklearn.linear_model import Ridge
ridgereg = Ridge(alpha=0, normalize=True)
ridgereg.fit(X_train, y_train)
y_pred = ridgereg.predict(X_test)
print(y_pred)
# calculate R^2 value, MAE, MSE, RMSE

from sklearn import metrics
print("R-Square Value",r2_score(y_test,y_pred))
print("\n")
print ("mean_absolute_error :",metrics.mean_absolute_error(y_test, y_pred))
print("\n")
print ("mean_squared_error : ",metrics.mean_squared_error(y_test, y_pred))
print("\n")
print ("root_mean_squared_error : ",np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# try alpha=0.1
ridgereg = Ridge(alpha=0.1, normalize=True)
ridgereg.fit(X_train, y_train)
y_pred2 = ridgereg.predict(X_test)
print(y_pred2)
# calculate R^2 value, MAE, MSE, RMSE

from sklearn import metrics
print("R-Square Value",r2_score(y_test,y_pred2))
print("\n")
print ("mean_absolute_error :",metrics.mean_absolute_error(y_test, y_pred2))
print("\n")
print ("mean_squared_error : ",metrics.mean_squared_error(y_test, y_pred2))
print("\n")
print ("root_mean_squared_error : ",np.sqrt(metrics.mean_squared_error(y_test, y_pred2)))
print(ridgereg.coef_)

# create an array of alpha values
alpha_range = 10.**np.arange(-2, 3)
alpha_range
#ar1=array([1.e-02, 1.e-01, 1.e+00, 1.e+01, 1.e+02])
# select the best alpha with RidgeCV
from sklearn.linear_model import RidgeCV
ridgeregcv = RidgeCV(alphas=alpha_range, normalize=True, scoring='neg_mean_squared_error')
ridgeregcv.fit(X_train, y_train)
print(ridgeregcv.alpha_)
# predict method uses the best alpha value
y_pred3 = ridgeregcv.predict(X_test)
print(y_pred3)

# calculate R^2 value, MAE, MSE, RMSE

from sklearn import metrics
print("R-Square Value",r2_score(y_test,y_pred3))
print("\n")
print ("mean_absolute_error :",metrics.mean_absolute_error(y_test, y_pred3))
print("\n")
print ("mean_squared_error : ",metrics.mean_squared_error(y_test, y_pred3))
print("\n")
print ("root_mean_squared_error : ",np.sqrt(metrics.mean_squared_error(y_test, y_pred3)))

# examing the coefficients
print(ridgereg.coef_)

#Lasso model
# try alpha=0.001 and examine coefficients
from sklearn.linear_model import Lasso
lassoreg = Lasso(alpha=0.001, normalize=True)
lassoreg.fit(X_train, y_train)
print(lassoreg.coef_)

# try alpha=0.01 and examine coefficients
lassoreg2 = Lasso(alpha=0.01, normalize=True)
lassoreg2.fit(X_train, y_train)
print(lassoreg2.coef_)

# calculate RMSE (for alpha=0.01)
y_pred4 = lassoreg.predict(X_test)
# calculate MAE, MSE, RMSE
# calculate R^2 value, MAE, MSE, RMSE

from sklearn import metrics
print("R-Square Value",r2_score(y_test,y_pred4))
print("\n")
print ("mean_absolute_error :",metrics.mean_absolute_error(y_test, y_pred4))
print("\n")
print ("mean_squared_error : ",metrics.mean_squared_error(y_test, y_pred4))
print("\n")
print ("root_mean_squared_error : ",np.sqrt(metrics.mean_squared_error(y_test, y_pred4)))

# select the best alpha with LassoCV
from sklearn.linear_model import LassoCV
lassoregcv = LassoCV(n_alphas=100, normalize=True, random_state=1)
lassoregcv.fit(X_train, y_train)
print('alpha : ',lassoregcv.alpha_)

# examine the coefficients
print(lassoregcv.coef_)

# predict method uses the best alpha value
y_pred5 = lassoregcv.predict(X_test)
# calculate R^2 value, MAE, MSE, RMSE

from sklearn import metrics
print("R-Square Value",r2_score(y_test,y_pred5))
print("\n")
print ("mean_absolute_error :",metrics.mean_absolute_error(y_test, y_pred5))
print("\n")
print ("mean_squared_error : ",metrics.mean_squared_error(y_test, y_pred5))
print("\n")
print ("root_mean_squared_error : ",np.sqrt(metrics.mean_squared_error(y_test, y_pred5)))

