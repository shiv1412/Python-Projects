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
import scipy.stats as st
import scipy.special as sp

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

# z test
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import math
import scipy.stats as st
import scipy.special as sp
n = 100
h = 61
q = .5
xbar = float(h)/ n
z = (xbar - q) * np.sqrt(n / (q * (1-q)))
print(z)
pval = 2*(1-st.norm.cdf(z))
print(pval)

# Annova test
import pandas as pd
d = pd.read_csv("https://reneshbedre.github.io/myfiles/anova/onewayanova.txt", sep="\t")
d.boxplot(column=['A','B','C', 'D'],grid=False)
import scipy.stats as stats
fvalue,pvalue = stats.f_oneway(d['A'],d['B'],d['C'],d['D'])
print(fvalue,pvalue)
import statsmodels.api as sm
from statsmodels.formula.api import ols
d_melt = pd.melt(d.reset_index(),id_vars=['index'],value_vars=['A','B','C','D'])
d_melt.columns = ['index','treatments','value']
model = ols('value ~ C(treatments)',data=d_melt).fit()
anova_table = sm.stats.anova_lm(model,typ=2)
anova_table
from statsmodels.stats.multicomp import pairwise_tukeyhsd
m_comp = pairwise_tukeyhsd(endog=d_melt['value'], groups=d_melt['treatments'], alpha=0.05)
print(m_comp)

w, pvalue = stats.shapiro(model.resid)
print(w, pvalue)

# Two way Annova
import pandas as pd
import seaborn as sns
d = pd.read_csv("https://reneshbedre.github.io/myfiles/anova/twowayanova.txt", sep="\t")
d_melt = pd.melt(d, id_vars=['Genotype'], value_vars=['1_year', '2_year', '3_year'])
d_melt.columns = ['Genotype', 'years', 'value']
sns.boxplot(x="Genotype", y="value", hue="years", data=d_melt, palette="Set3") 
import statsmodels.api as sm
from statsmodels.formula.api import ols
model = ols('value ~ C(Genotype) + C(years) + C(Genotype):C(years)', data=d_melt).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
anova_table


#chi-squared test
import numpy as np
a1 = [6, 4, 5, 10]
a2 = [8, 5, 3, 3]
a3 = [5, 4, 8, 4]
a4 = [4, 11, 7, 13]
a5 = [5, 8, 7, 6]
a6 = [7, 3, 5, 9]
dice = np.array([a1, a2, a3, a4, a5, a6])
from scipy import stats

stats.chi2_contingency(dice)
chi2_stat, p_val, dof, ex = stats.chi2_contingency(dice)
print("===Chi2 Stat===")
print(chi2_stat)
print("\n")
print("===Degrees of Freedom===")
print(dof)
print("\n")
print("===P-Value===")
print(p_val)
print("\n")
print("===Contingency Table===")
print(ex)
r1 = np.random.randint(1,7,1000)
r2 = np.random.randint(1,7,1000)
r3 = np.random.randint(1,7,1000)
r4 = np.random.randint(1,7,1000)
r5 = np.random.randint(1,7,1000)

unique, counts1 = np.unique(r1, return_counts=True)
unique, counts2 = np.unique(r2, return_counts=True)
unique, counts3 = np.unique(r3, return_counts=True)
unique, counts4 = np.unique(r4, return_counts=True)
unique, counts5 = np.unique(r5, return_counts=True)

dice = np.array([counts1, counts2, counts3, counts4, counts5])
chi2_stat, p_val, dof, ex = stats.chi2_contingency(dice)

my_rolls_expected = [46.5, 46.5, 46.5, 46.5, 46.5, 46.5]
my_rolls_actual =  [59, 63, 37, 38, 32, 50]
st1 = stats.chisquare(my_rolls_actual, my_rolls_expected)
print(st1)
opp_rolls_expected = [50.5,50.5,50.5,50.5,50.5,50.5]
opp_rolls_actual =  [39,39,46,54,53,72]
st2 = stats.chisquare(opp_rolls_actual, opp_rolls_expected)
print(st2)

