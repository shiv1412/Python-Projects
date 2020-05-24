# -*- coding: utf-8 -*-
"""
Created on Sat May 23 17:49:01 2020

@author: sharm
"""

input=["+1A", "+3E", "-1A", "+4F", "+1A", "-3E"]
dict = {}
for e1 in input:
    if e1 not in dict:
        dict[e1]= 1
    else:
        dict[e1]+=1
i = 0
k=""
for el in dict:
    if dict[e1]>i:
        i=dict[e1]
        k=e1
    print(k)