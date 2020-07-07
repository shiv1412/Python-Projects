# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 09:34:15 2020

@author: sharm
"""

import math
import matplotlib.pyplot as plt
import numpy as np

#sigmoid function
def sigmoid(x):
    a = []
    for i in x:
        a.append(1/(1+math.exp(-1)))
    return a
#hyperbolic tangent function

def tanh(x,derivative=False):
    if (derivative ==True):
        return (1-(x ** 2))
    return np.tanh(x)

#RELU

def re(x):
    b = []
    for i in x:
        if i<0:
            b.append(0)
        else:
            b.append(i)
    return b
# leaky RElu

def lr(x):
    b =[]
    for i in x:
        if i<0:
           b.append(i/10)
        else:
           b.append(i)
    return b

# visualization of graphs

x =np.arange(-3.,3.,0.1)
sig = sigmoid(x)
tanh = tanh(x)
relu = re(x)
leaky_relu = lr(x)
swish = sig*x

line_1, = plt.plot(x,sig, label='Sigmoid')
line_2, = plt.plot(x,tanh, label='Tanh')
line_3, = plt.plot(x,relu, label='ReLU')
line_4, = plt.plot(x,leaky_relu, label='Leaky ReLU')
line_5, = plt.plot(x,swish, label='Swish')
plt.legend(handles=[line_1, line_2, line_3, line_4, line_5])
plt.axhline(y=0, color='k')
plt.axvline(x=0, color='k')
plt.show()