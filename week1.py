# -*- coding: utf-8 -*-
"""
Created on Tue May 26 18:53:46 2020

@author: sharma shivani
"""
#required pacakges import
import re
import string
import numpy as np 
import random
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from plotly import graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff
from collections import Counter
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import nltk
from nltk.corpus import stopwords
from tqdm import tqdm
import os
import nltk
import spacy
import random
from spacy.util import compounding
from spacy.util import minibatch
import warnings
warnings.filterwarnings("ignore")
import os

#data set reading
for dirname, _, filenames in os.walk('D:/Data Mining/tweet-sentiment-extraction-Week1'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# color generation function        
def random_colours(number_of_colors):
    colors = []
    for i in range(number_of_colors):
        colors.append("#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]))
    return colors
#reading test and training data
train = pd.read_csv('D:/Data Mining/tweet-sentiment-extraction-Week1/train.csv')
test = pd.read_csv('D:/Data Mining/tweet-sentiment-extraction-Week1/test.csv')
ss = pd.read_csv('D:/Data Mining/tweet-sentiment-extraction-Week1/sample_submission.csv')
print(train.shape)
print(test.shape)

#EDA
#1. Differntiating words as per the type of statement i.e negative,positive or neutral
train.info()
train.dropna(inplace=True)
test.info()
train.head()
train.describe()
plt.figure(figsize=(12,6))
sns.countplot(x='sentiment',data=train)
hist_data = [train['Num_words_ST'],train['Num_word_text']]
group_labels = ['Selected_Text', 'Text']

# Create distplot with custom bin_size
fig = ff.create_distplot(hist_data, group_labels,show_curve=False)
fig.update_layout(title_text='Distribution of Number Of words')
fig.update_layout(
    autosize=False,
    width=900,
    height=700,
    paper_bgcolor="LightSteelBlue",
)
plt.figure()
fig.show()
# KErnal dstribution of difference of number of words
plt.figure(figsize=(12,6))
p1=sns.kdeplot(train['Num_words_ST'], shade=True, color="r").set_title('Kernel Distribution of Number Of words')
p1=sns.kdeplot(train['Num_word_text'], shade=True, color="b")

plt.figure(figsize=(12,6))
p1=sns.kdeplot(train[train['sentiment']=='positive']['difference_in_words'], shade=True, color="b").set_title('Kernel Distribution of Difference in Number Of words')
p2=sns.kdeplot(train[train['sentiment']=='negative']['difference_in_words'], shade=True, color="r")

