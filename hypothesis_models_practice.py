# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 10:56:54 2020

@author: sharm
"""
# Hypothesis test: one sample T test
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import math

np.random.seed(60)
population_ages1 = stats.poisson.rvs(loc=18,mu=35,size=15000)
population_ages2 = stats.poisson.rvs(loc=18,mu=10,size=15000)
population_ages = np.concatenate((population_ages1,population_ages2))

minnetosa_ages1 = stats.poisson.rvs(loc=18,mu=30,size=30)
minnetosa_ages2 = stats.poisson.rvs(loc=18,mu=10,size=20)
minnetosa_ages = np.concatenate((minnetosa_ages1,minnetosa_ages2))


print(population_ages.mean())
print(minnetosa_ages.mean())

st1 = stats.ttest_1samp(a=minnetosa_ages,popmean=population_ages.mean())
st2 = stats.t.ppf(q=0.025,df=49)
print(st1)
print(st2)

st3 = stats.t.ppf(q=0.025,df=49)
st4 = stats.t.ppf(q=0.975,df=49)
print(st3)
print(st4)

st5 = stats.t.cdf(x=-2.5742,df=49)*2

sigma = minnetosa_ages.std()/math.sqrt(50)
st7 = stats.t.interval(0.95,df=49,loc=minnetosa_ages.mean(),scale=sigma)

print(st5)
print(sigma)
print(st7)

#two sample t test

np.random.seed(12)
wisconsin_ages1 = stats.poisson.rvs(loc=18, mu=33, size=30)
wisconsin_ages2 = stats.poisson.rvs(loc=18, mu=13, size=20)
wisconsin_ages = np.concatenate((wisconsin_ages1, wisconsin_ages2))

print( wisconsin_ages.mean() )

str8= stats.ttest_ind(a=minnetosa_ages,
                b=wisconsin_ages,equal_var=False)
print(str8)

# paired t test

np.random.seed(10)
before = stats.norm.rvs(scale=30,loc=250,size=100)
after = before + stats.norm.rvs(scale=5, loc=250, size=100)

weight_df = pd.DataFrame({"weight_before":before,
                          "weight_after":after,
                          "weight_change":after-before})
print(weight_df.describe())
p_test = stats.ttest_rel(a=before, b=after)
print(p_test)


#Type 1 and type 2 errors
plt.figure(figsize=(12,10))
plt.fill_between(x=np.arange(-4,-2,0.01),
                 y1 = stats.norm.pdf(np.arange(-4,-2,0.01)),
                 facecolor='red',
                 alpha=0.35)

plt.fill_between(x=np.arange(-2,2,0.01), 
                 y1= stats.norm.pdf(np.arange(-2,2,0.01)) ,
                 facecolor='white',
                 alpha=0.35)

plt.fill_between(x=np.arange(2,4,0.01), 
                 y1= stats.norm.pdf(np.arange(2,4,0.01)) ,
                 facecolor='red',
                 alpha=0.5)

plt.fill_between(x=np.arange(-4,-2,0.01), 
                 y1= stats.norm.pdf(np.arange(-4,-2,0.01),loc=3, scale=2) ,
                 facecolor='white',
                 alpha=0.35)

plt.fill_between(x=np.arange(-2,2,0.01), 
                 y1= stats.norm.pdf(np.arange(-2,2,0.01),loc=3, scale=2) ,
                 facecolor='blue',
                 alpha=0.35)

plt.fill_between(x=np.arange(2,10,0.01), 
                 y1= stats.norm.pdf(np.arange(2,10,0.01),loc=3, scale=2),
                 facecolor='white',
                 alpha=0.35)

plt.text(x=-0.8, y=0.15, s= "Null Hypothesis")
plt.text(x=2.5, y=0.13, s= "Alternative")
plt.text(x=2.1, y=0.01, s= "Type 1 Error")
plt.text(x=-3.2, y=0.01, s= "Type 1 Error")
plt.text(x=0, y=0.02, s= "Type 2 Error")