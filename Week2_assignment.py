# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 16:05:52 2020

@author: sharm
"""
# Importing all the required libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn import linear_model
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.preprocessing import StandardScaler
def main():
        log_func=log_model_building()

def log_model_building():
    # reading dataset files
        training_dataset = pd.read_csv('D:/Predictive Course/fashion/fashion_train.csv')
        testing_dataset = pd.read_csv('D:/Predictive Course/fashion/fashion_test.csv')
        label = {
            0: 'T-shirt/top',
            1: 'Trouser',
            2: 'Pullover',
            3: 'Dress',
            4: 'Coat',
            5: 'Sandal',
            6: 'Shirt',
            7: 'Sneaker',
            8: 'Bag',
            9: 'Ankle boot',
            }
        # print(training_dataset.head(10))
        # testing_dataset.head(10)
        # splitting the datasets into train and test parts for labels and features
        y_train = training_dataset['label']
        y_test = testing_dataset['label']
        X_train = training_dataset.drop('label',axis=1)
        X_test = testing_dataset.drop('label',axis=1)
        y_train= y_train.values.reshape(-1,1)
        y_test= y_test.values.reshape(-1,1)
        scalar = StandardScaler()
        scalar.fit(y_train)
        scalar.fit(y_test)
        # Logistic regression model logic
        mul_lr = linear_model.LogisticRegression(multi_class='multinomial', 
                            solver='saga',penalty='l2',max_iter=100)
        fit1 = mul_lr.fit(X_train, y_train)
        x_p1=mul_lr.predict_proba(X_train)
        x_p2=mul_lr.predict_proba(X_test)
        predicted_val = mul_lr.predict(X_train)
        predicted_test = mul_lr.predict(X_test)
        print(predicted_val)
        # calcuation of accuracy for train and test datasets
        print ("Training dataset Accuracy :: ", metrics.accuracy_score(y_train, mul_lr.predict(X_train)))
        print ("Testing dataset Accuracy :: ", metrics.accuracy_score(y_test, mul_lr.predict(X_test)))
       #calculation of scores
        score1=mul_lr.score(X_train,y_train)
        score2=mul_lr.score(X_test,y_test)
        # confusion matrix calculation
        confusion_matrix(y_test,predicted_test)
        cm1 = confusion_matrix(y_test,predicted_test)
        fig,ax = plt.subplots(figsize=(6,6))
        ax.imshow(cm1)
        plt.show()
        # plotting heatmap for the confusion matrix
        plt.figure(figsize=(9,9))
        sns.heatmap(cm1, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
        plt.ylabel('Actual values values');
        plt.xlabel('Predicted values');
        title1 = 'Score for testing dataset: {0}'.format(score2)
        plt.title(title1, size = 15)
        
        confusion_matrix(y_train,predicted_val)
        cm2 = confusion_matrix(y_train,predicted_val)
        fig,ax = plt.subplots(figsize=(6,6))
        ax.imshow(cm2)
        plt.show()
        # plotting heatmap for the confusion matrix
        plt.figure(figsize=(9,9))
        sns.heatmap(cm2, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
        plt.ylabel('Actual values values');
        plt.xlabel('Predicted values');
        title2 = 'Score for training dataset: {0}'.format(score1)
        plt.title(title2, size = 15)
        
        #softmax function calculation for converting the obtained predicted probablities intointo probability distribution over 10 digits
        predicted_val -= np.max(predicted_val)
        sm_train = (np.exp(predicted_val).T / np.sum(np.exp(predicted_val))).T
        sm_test = (np.exp(predicted_test).T / np.sum(np.exp(predicted_test))).T
       # plotting softmax results 
        plt.plot(sm_train)
        plt.plot(sm_test)
        print(sm_train)
        print(sm_test)


if __name__ == '__main__':
            main()