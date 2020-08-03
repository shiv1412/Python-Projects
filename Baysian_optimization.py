# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 12:01:22 2020

@author: sharm
"""

from math import sin
from math import pi
from numpy import arange
from numpy import vstack
from numpy import argmax
from numpy import asarray
from numpy.random import normal
from numpy.random import random
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from warnings import catch_warnings
from warnings import simplefilter
from matplotlib import pyplot

# objective function
def objective(x,noise=0.1):
    noise=normal(loc=0,scale=noise)
    return(x**2 * sin(5*pi*x)**6) + noise
# surrogate function or approximation function
def surrogate(model,X):
    with catch_warnings():
        simplefilter("ignore")
        return model.predict(X,return_std=True)
# acq function
def acquisition(X, Xsamples, model):
    yhat,_=surrogate(model,X)
    best = max(yhat)
    mu,std = surrogate(model,Xsamples)
    mu=mu[:,0]
    probs = norm.cdf((mu - best) / (std+1E-9))
    return probs

def opt_acquisition(X,y,model):
    Xsamples=random(100)
    Xsamples=Xsamples.reshape(len(Xsamples),1)
    scores = acquisition(X, Xsamples, model)
    # locate the index of the largest scores
    ix = argmax(scores)
    return Xsamples[ix, 0]

def plot(X,y,model):
    pyplot.scatter(X,y)
    Xsamples=asarray(arange(0,1,0.001))
    Xsamples = Xsamples.reshape(len(Xsamples),1)
    ysamples,_=surrogate(model,Xsamples)
    pyplot.plot(Xsamples,ysamples)
    pyplot.show()

X = random(100)
y = asarray([objective(x) for x in X])
X = X.reshape(len(X),1)
y = y.reshape(len(y),1)

model = GaussianProcessRegressor()

model.fit(X,y)

plot(X,y,model)

for i in range(100):
    x = opt_acquisition(X, y, model)
    actual = objective(X)
    est,_=surrogate(model, [[X]])
    print(x,est,actual)
    X = vstack((X, [[x]]))
    y = vstack((y, [[actual]]))
    model.fit(X,y)

    
plot(X,y,model)
ix = argmax(y)
print('Best Result: x=%.3f, y=%.3f' % (X[ix], y[ix]))

