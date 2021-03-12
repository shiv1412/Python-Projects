# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 16:18:54 2020

@author: sharm
"""
# importing libraries
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from keras.preprocessing.text import Tokenizer
from tokenizer import tokenizer
from nltk.tokenize import RegexpTokenizer
import nltk


# creating dataframes for the provided data and reading it
colum_headers = ['sentence', 'class']
row_values = []

row_values = [['Your limitation its only your imagination', 'Inspiring'], 
        ['Push yourself because no one else is going to do it for you', 'Inspiring'],
        ['Sometimes later becomes never do it now', 'Inspiring'],
        ['Great things never come from comfort zones', 'Inspiring'],
        ["Its great to be yourself", 'Dull'],
        ['A goal is a dream with a plan', 'Dull']]

training_dataset = pd.DataFrame(row_values, columns=colum_headers)
training_dataset



from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
    
stop_words = set(stopwords.words('english')) 
  
word_tokens = word_tokenize(training_dataset['sentence']) 
  
filtered_sentence = [w for w in word_tokens if not w in stop_words] 
  
filtered_sentence = [] 
  
for w in word_tokens: 
    if w not in stop_words: 
        filtered_sentence.append(w) 
  
print(word_tokens) 
print(filtered_sentence)







# interatiing over the sentences and creating vectors for 
#the words for categorizing the data as inspiring
inspiring_document = [row_values['sentence'] for index,row_values in training_dataset.iterrows() if row_values['class'] == 'Inspiring']
vector_for_inspiring = CountVectorizer()
X_inspiring = vector_for_inspiring.fit_transform(inspiring_document)
term_document_matrix_inspiring = pd.DataFrame(X_inspiring.toarray(), columns=vector_for_inspiring.get_feature_names())
print(term_document_matrix_inspiring)

# interatiing over the sentences and creating vectors for 
#the words for categorizing the data as dull
dull_document = [row_values['sentence'] for index,row_values in training_dataset.iterrows() if row_values['class'] == 'Dull']
vector_for_dull = CountVectorizer()
X_dull = vector_for_dull.fit_transform(dull_document)
term_document_matrix_dull = pd.DataFrame(X_dull.toarray(), columns=vector_for_dull.get_feature_names())
print(term_document_matrix_dull)


# creating word wise listing for inspiring class and calculating the frequencies of words
word_list_inspiring = vector_for_inspiring.get_feature_names()    
count_list_inspiring = X_inspiring.toarray().sum(axis=0) 
freq_inspiring = dict(zip(word_list_inspiring,count_list_inspiring))
freq_inspiring

# creating word wise listing for dull class and calculating the frequencies of words
word_list_dull = vector_for_dull.get_feature_names();    
count_list_dull = X_dull.toarray().sum(axis=0) 
freq_dull = dict(zip(word_list_dull,count_list_dull))
freq_dull

# finding out probablities for each word in  inspiring class
prob_insp = []
for word,count in zip(word_list_inspiring,count_list_inspiring):
    prob_insp.append(count/len(word_list_inspiring))
dict(zip(word_list_inspiring,prob_insp))


# finding out probabilites for each word in dull class
prob_dull = []
for word,count in zip(word_list_dull,count_list_dull):
    prob_dull.append(count/len(word_list_dull))
dict(zip(word_list_dull,prob_dull))



# creating a document for all rhe sentences in the training dataset
document = [row_values['sentence'] for index,row_values in training_dataset.iterrows()]
# creating vector for the whole dataset and storing it in a document
vector = CountVectorizer()
X = vector.fit_transform(document)
print(document)
print(X)


# caculation of total features for the above vector created
total_features = len(vector.get_feature_names())
print(total_features)

# calculating total count of features for both inspiring and dull
total_counts_features_inspiring = count_list_inspiring.sum(axis=0)
total_counts_features_dull = count_list_dull.sum(axis=0)
print(total_counts_features_inspiring)
print(total_counts_features_dull)


# coming to the processing of the sentence to remove punctuation before doing the classification of sentence Dream it.Wish it.Do it. into inspiring/dull
from nltk.tokenize import word_tokenize
from string import punctuation
def strip_punctuation(s):
    return ''.join(c for c in s if c not in punctuation)
test_sentence = 'Dream it. Wish it. Do it.'
new_sentence = strip_punctuation(test_sentence)
print (new_sentence)
new_word_list = word_tokenize(new_sentence)
print(new_word_list)


# finding out probability of the sentence if it is inspiring by using the frequnecy
# calculated on training data set storing them in a dictionary
prob_inspiring_new_sentence = []
for word in new_word_list:
    if word in freq_inspiring.keys():
        count = freq_inspiring[word]
    else:
        count = 0
    prob_inspiring_new_sentence.append((count + 1)/
                                       (total_counts_features_inspiring + total_features))
print(dict(zip(new_word_list,prob_inspiring_new_sentence)))


# multiplying the probabilities of all the words
print(0.0149*0.0447*0.0149*0.0149)


# multiplying the probability with the probability of occurance of the category which is 0.5 for each class i.e 
#either inspiring or dull
print(1.478653203e-07*0.5)


# finding out probability of the sentence if it is dull by using the frequnecy
# calculated on training data set storing them in a dictioanry
prob_dull_new_sentence = []
for word in new_word_list:
    if word in freq_dull.keys():
        count = freq_dull[word]
    else:
        count = 0
    prob_dull_new_sentence.append((count + 1)/(total_counts_features_dull + 
                                               total_features))
print(dict(zip(new_word_list,prob_dull_new_sentence)))


print(0.0227*0.0227*0.227*0.227)


# multiplying the probability obtained above with the probability of occurance of the category which is 0.5 foreach classi.e 
#either inspiring or dull
print(2.6552378410000006e-05*0.5)
