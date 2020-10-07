# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 12:46:36 2020

@author: sharm
"""

import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
#from adspy_shared_utilities import plot_class_regions_for_classifier_subplot
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

#def __init__(self,training_set,testing_set,y_train,X_test,X_train,y_test,y_predicted):
training_set = pd.read_csv('D:/Predictive Course/MNIST-data/mnist_train.csv')
training_set=training_set
testing_set= pd.read_csv('D:/Predictive Course/MNIST-data/mnist_test.csv')
testing_set=testing_set
y_train = training_set['label']
#y_train=y_train
y_test = testing_set['label']
#y_test=y_test
X_train = training_set.drop('label',axis=1)
#X_train=X_train
X_test = testing_set.drop('label',axis=1)
#X_test=X_test
y_train= y_train.values.reshape(-1,1)
y_test= y_test.values.reshape(-1,1)
scalar = StandardScaler()
scalar.fit(y_train)
scalar.fit(y_test)
segment_labeling = {
0: 'a',
1: 'b',
2: 'c',
3: 'd',
4: 'e',
5: 'f',
6: 'g'
}
y_predicted = np.array([np.median(y_train)]*len(y_train))


#def grad_boosting_algo(self,inp_param=None):
# if inp_param is None:
#         y_inputs = training_set,testing_set,y_test,X_test,y_train,X_train,y_predicted
        
grad_boost_classifier = GradientBoostingClassifier(learning_rate=0.01,max_depth=2,random_state=0)
    #title = 'Gradient boosting binary dataset'
   # plot_class_regions_for_classifier_subplot(grad_boost_classifier,X_train,y_train,X_test,y_test,title)
grad_boost_classifier.fit(X_train,y_train)

predicted_values = grad_boost_classifier.predict(X_test)

accuracy_train_set = grad_boost_classifier.score(X_train,y_train)
accuracy_test_set = grad_boost_classifier.score(X_test,y_test)
print(accuracy_train_set)
print(accuracy_test_set)

confusion_matrix(y_test,predicted_values)
cm = confusion_matrix(y_test,predicted_values)
fig,ax = plt.subplots(figsize=(6,6))
ax.imshow(cm)
plt.show()
# plotting heatmap for the confusion matrix
plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual values values');
plt.xlabel('Predicted values');
title1 = 'Score for testing dataset: {0}'.format(accuracy_test_set)
plt.title(title1, size = 15)

# def main():
#     grad_boosting_algo()
    

# if __name__ == '__main__':
#             main()
            

# def loss_func(self,y_inputs=None):
#     if y_inputs is None:
#         y_inputs = self.training_set,self.testing_set,self.y_test,self.X_test,self.y_train,self.X_train,self.y_predicted
#         for i in len(self.training_set):
#             loss = 0
#             loss = ((self.y_train - y_predicted)**2)/2
#             return loss
    
# def pseudo_residuals(self,inputs=None):
#     if inputs is None:
#         inputs = self.training_set,self.testing_set,self.y_test,self.X_test,self.y_train,self.X_train,self.y_predicted
#         for i in len(self.training_set):
#             self.loss(self.y_train[0],self.y_predicted[0])  
#             for i in range(len(self.training_set)):
#                 losses = []
#                 losses.append(self.loss_func(self.y_train[0], self.y_predicted[0]+i/(len(self.training_set)/2)))
#                 grad_cal = (self.loss_func(self.y_train[0], self.y_predicted[0]+1) - self.loss_func(self.y_train[0], self.y_predicted[0])) / self.y_predicted+1
#                 return grad_cal
#     pseudo_residuals()
    
# def grad_loss_func(self,dataparameters= None):
#     if dataparameters is None:
#         dataparameters = self.training_set,self.testing_set,self.y_test,self.X_test,self.y_train,self.X_train,self.y_predicted
#         grad_loss = (-(self.y_train - self.y_predicted)
#         return grad_loss
    
# def learner_func(self,param= None):
#     if param is None:
#         param = self.training_set,self.testing_set,self.y_test,self.X_test,self.y_train,self.X_train,self.y_predicted
#         # fitting a base learner  i.e tree to pseudo residual 
#         regressing = DecisionTreeRegressor(max_depth=1)
#         regressing.fit(self.X_train,self.pseudo_residuals)
#         regressing.tree_.feature
#         thresh_val= regressing.tree_.threshold
#         branch_1_index = np.where(self.X_train[:,len(self.X_train)/2]) >= thresh_val
#         branch_2_index = np.where(self.X_train[:,len(self.X_train)/2]) < thresh_val
#         branch1_op = branch_1_index.mean() 
#         branch2_op = branch_2_index.mean()
#         self.grad_loss_func(self.grad_loss_func,[ branch1_op if ix in list(branch_1_index)[0] else branch2_op for ix in range(len(self.X_train))]).mean()
#         np.unique(regressing.predict(self.X_train))
        
#         leaf1_loss = [] 
#         for i in np.arange(0, len(self.training_set), 0.01):      
#             leaf1_loss.append(self.grad_loss_func(self.y_train[branch_1_index], i * regressing.predict(self.X_train[branch_1_index]) + self.y_predicted[branch_1_index]).mean())
#             np.argmin(leaf1_loss)
            
            
#         leaf2_loss = [] 
#         for i in np.arange(0, len(self.training_set), 0.01):      
#             leaf2_loss.append(self.grad_loss_func(self.y_train[branch_2_index], i * regressing.predict(self.X_train[branch_1_index]) + self.y_predicted[branch_2_index]).mean())
#             np.argmin(leaf2_loss)
        
        
#         # update the model step 
#         func1 = self.y_predicted + regressing.predict(self.X_train)
#         self.grad_loss_func(self.y_train,func1).mean()
        
        
#         residuals = -self.grad_loss_func(self.y_train, func1) 
#         regressing.fit(self.X_train, residuals) 
#         func2 = f1 + regressing.predict(self.X_train) 
#         self.grad_loss_func(self.y_train, func2).mean()

# def absolute_loss_gradient(self,in_val=None): 
#     return np.array([ -1 if self.y_train > self.y_predicted else 1 for self.y_trainy, self.y_predicted in np.c_[self.y_train, self.y_predicted] ])

# def absolut_loss_value(self,y_param=None): 
#     if in_param is None:
#         in_param = self.training_set,self.testing_set,self.y_test,self.X_test,self.y_train,self.X_train,self.y_predicted
#         absolute_loss_value= np.abs(self.y_train - self.y_predicted)
#         return absolute_loss_value

# def gradient_boost_mae(self, M,in_param=None): 
#     if in_param is None:
#         in_param = self.training_set,self.testing_set,self.y_test,self.X_test,self.y_train,self.X_train,self.y_predicted
#         M = len(self.training_set)
#         print(loss_func(self.y_train,self.y_predicted).mean())
        
#         for i in range(M): 
#             residual_value = -absolute_loss_gradient(self.y_train, self.y_predicted) 
#             return residual_value
        
#         regressor = DecisionTreeRegressor(max_depth=1) 
#         regressor.fit(self.X_train, residuals) 
#         find_features = regressor.tree_.find_features[0] 
#         threshold_value = regressor.tree_.threshold_value[0] 
#         first_leaf_index = np.where(self.X_train[:,find_features] >= threshold_value) 
#         second_leaf_index = np.where(self.X_train[:,find_features] < threshold_value)   
#         first_leaf_op = residual_value[first_leaf_index].mean() 
#         leaf2_output = residual_value[leaf2_index].mean() 
        
#         first_leaf_loss_value = [] 
#         for i in np.arange(0, len(self.training_set)/2, 0.01): 
#            first_leaf_loss_value.append(self.loss_func(self.y_train[first_leaf_index], self.y_predicted[first_leaf_index] + i * regressor.predict(self.X_train[first_leaf_index])).mean())
#         leaf1_gamma_val = (np.argmin(first_leaf_loss_value) /(len(self.training_set/2)))
       
#         second_leaf_loss_value = [] 
#         for i in np.arange(0, len(self.training_set)/2, 0.01): 
#            second_leaf_loss_value.append(self.loss_func(self.y_train[second_leaf_index], self.y_predicted[second_leaf_index] + i * regressor.predict(self.X_train[second_leaf_index])).mean())
#         leaf2_gamma_val = (np.argmin(second_leaf_loss_value /(len(self.training_set/2)))
       
                           
#         for i in range(len(self.X_train)):
#             if i in list(first_leaf_index[0]):
#                 predicted_value = first_leaf_op * leaf1_gamma_val 
#             else:
#                 predicted_value= second_leaf_op * leaf2_gamma_val 
                           
                
#         self.y_predicted = self.y_predicted + predicted_value 
#         print(absolut_loss_value(self.y_train, self.y_predicted).mean())


# def gradient_boost_mse(self,X, y, M, learning_rate,data_param=None):        
# regressors = [] 
#     y_hat = np.array([y.mean()]*len(y)) 
#     f0 = y_hat 
#     print(compute_loss(y, y_hat).mean()) 
#     for i in range(M): 
#         residuals = -loss_gradient(y, y_hat) 
#         regressor = DecisionTreeRegressor(max_depth=1) 
#         regressor.fit(X, residuals) 
#         regressors.append(regressor) 
#         predictions = regressor.predict(X) y_hat = y_hat + learning_rate * predictions 
#         print(compute_loss(y, y_hat).mean()) 
#     return regressors, f0
# def gradient_boost_mse_predict(regressors, f0, X, learning_rate):
#     y_hat = np.array([f0[0]]*len(X)) 
#     for regressor in regressors: 
#         y_hat = y_hat + learning_rate * regressor.predict(X) 
#     return y_hat



# def activation_function_logic():
nums = [0,1,2,3,4,5,6,7,8,9]
a = [0,1,0,0,1,0,0,0,0,0]
b = [0,0,0,0,0,1,1,0,0,0]
c = [0,0,1,1,1,1,1,1,1,1]
d = [0,1,0,0,1,0,0,1,0,1]
e = [0,1,0,1,1,1,0,1,0,1]
f = [0,1,1,1,0,0,0,1,0,0]
g = [1,1,0,0,0,0,0,1,0,0]

A = [0,0,0,0,0,0,0,0,1,1]
B = []    
        
#     for i in nums[]:
#         for j in a[]:
#             for k in b
        


