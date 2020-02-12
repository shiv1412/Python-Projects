import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams.update({'font.size':12})

from sklearn.datasets import load_boston
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge

boston=load_boston()
boston_df=pd.DataFrame(boston.data,columns=boston.feature_names)
#print boston_df.info()
boston_df['Price']=boston.target
#print boston_df.head(3)

newX=boston_df.drop('Price',axis=1)
#print newX[0:3]
newY=boston_df['Price']

#print type(newY)

X_train,X_test,Y_train,Y_test=train_test_split(newX,newY,test_size=0.3,random_state=3)

#print len(X_test),len(Y_test)

lr=LinearRegression()
lr.fit(X_train,Y_train)

rr=Ridge(alpha=0.01)

rr.fit(X_train,Y_train)

rr100=Ridge(alpha=100)
rr100.fit(X_train,Y_train)

train_score=lr.score(X_train,Y_train)
test_score=lr.score(X_test,Y_test)

Ridge_train_score=rr.score(X_train,Y_train)
Ridge_test_score=rr.score(X_test,Y_test)

Ridge_train_score100=rr100.score(X_train,Y_train)
Ridge_test_score100=rr100.score(X_test,Y_test)


#print "linear regression train score :", train_score
#print "linear regression test score ", test_score
#print "ridge regression train score low alpha", Ridge_train_score
#print "ridge regression test score with low alpha", Ridge_test_score
#print "ridge regression train score with high alpha", Ridge_train_score100
#print "Ridge regression test score with high alpha", Ridge_test_score100

plt.plot(rr.coef,alpha=0.7,linestyle='none',marker='*',markersize=5,color='red',label=r'Ridge; $\alpha=100$')

plt.plot(lr.coef_,alpha=0.4,linestyle='none',marker='o',markersize=7,color='green',label='Linear Regression')

plt.xlabel('coeficient Index',fontsize=16)
plt.ylabel('coefficient Magnitude',fontsize=16)
plt.legend(fontsize=13,loc=4)
plt.show()
