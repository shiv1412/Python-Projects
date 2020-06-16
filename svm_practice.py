# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 07:45:39 2020

@author: sharm
"""

import numpy as np
import cvxopt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix

class SVM:
    def fit(self,X,y):
        n_samples,n_features = X.shape
        k= np.zeros((n_samples,n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                k[i,j] = np.dot(X[i],X[j]) 

p = cvxopt.matrix(np.outer(y,y)*k)
# q = -1 (1xN)
q = cvxopt.matrix(np.ones(n_samples) * -1)
# A = y^T 
A = cvxopt.matrix(y, (1, n_samples))
# b = 0 
b = cvxopt.matrix(0.0)
# -1 (NxN)
G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
# 0 (1xN)
h = cvxopt.matrix(np.zeros(n_samples))
solution = cvxopt.solvers.qp(P, q, G, h, A, b)
# Lagrange multipliers
a = np.ravel(solution['x'])
# Lagrange have non zero lagrange multipliers
sv = a > 1e-5
ind = np.arange(len(a))[sv]
self.a = a[sv]
self.sv = X[sv]
self.sv_y = y[sv]
# Intercept
self.b = 0
for n in range(len(self.a)):
    self.b += self.sv_y[n]
    self.b -= np.sum(self.a * self.sv_y * K[ind[n], sv])
    self.b /= len(self.a)
# Weights
    self.w = np.zeros(n_features)
    for n in range(len(self.a)):
        self.w += self.a[n] * self.sv_y[n] * self.sv[n]
        
def project(self, X):
    return np.dot(X, self.w) + self.b
    
    
    def predict(self, X):
        return np.sign(self.project(X))
    
X, y = make_blobs(n_samples=250, centers=2,
                  random_state=0, cluster_std=0.60)
y[y == 0] = -1
tmp = np.ones(len(X))
y = tmp * y

plt.scatter(X[:, 0], X[:, 1], c=y, cmap='winter')
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
svm = SVM()
svm.fit(X_train, y_train)

svm = SVM()
svm.fit(X_train, y_train)

def f(x, w, b, c=0):
    return (-w[0] * x - b + c) / w[1]
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='winter')
# w.x + b = 0
a0 = -4; a1 = f(a0, svm.w, svm.b)
b0 = 4; b1 = f(b0, svm.w, svm.b)
plt.plot([a0,b0], [a1,b1], 'k')
# w.x + b = 1
a0 = -4; a1 = f(a0, svm.w, svm.b, 1)
b0 = 4; b1 = f(b0, svm.w, svm.b, 1)
plt.plot([a0,b0], [a1,b1], 'k--')
# w.x + b = -1
a0 = -4; a1 = f(a0, svm.w, svm.b, -1)
b0 = 4; b1 = f(b0, svm.w, svm.b, -1)
plt.plot([a0,b0], [a1,b1], 'k--')

y_pred = svm.predict(X_test)
confusion_matrix(y_test, y_pred)

svc = LinearSVC()
svc.fit(X_train, y_train)

plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='winter');
ax = plt.gca()
xlim = ax.get_xlim()
w = svc.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(xlim[0], xlim[1])
yy = a * xx - svc.intercept_[0] / w[1]
plt.plot(xx, yy)
yy = a * xx - (svc.intercept_[0] - 1) / w[1]
plt.plot(xx, yy, 'k--')
yy = a * xx - (svc.intercept_[0] + 1) / w[1]
plt.plot(xx, yy, 'k--')

y_pred = svc.predict(X_test)
confusion_matrix(y_test, y_pred)