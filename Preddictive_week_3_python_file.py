#%%

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

#%%

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

#%%

# Model training for learning rate 0.01
#pipeline = make_pipeline(StandardScaler(),                                                      
#                         GradientBoostingClassifier(learning_rate=0.01,max_depth=2,random_state=0))                   
#pipeline.fit(X_train, y_train)                                                                  
#pipeline.score(X_test, y_test)
grad_boost_classifier = GradientBoostingClassifier(learning_rate=0.01,max_depth=2,random_state=0)
grad_boost_classifier.fit(X_train,y_train)
#pipeline.predict(X_test)
predicted_values = grad_boost_classifier.predict(X_test)
accuracy_train_set = grad_boost_classifier.score(X_train,y_train)
accuracy_test_set = grad_boost_classifier.score(X_test,y_test)
print(accuracy_train_set)
print(accuracy_test_set)

#%%

# confusion matrix for above trained model
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

#%%

# Model training for learning rate 0.1
grad_boost_classifier = GradientBoostingClassifier(learning_rate=0.1,max_depth=2,random_state=0)
grad_boost_classifier.fit(X_train,y_train)
predicted_values = grad_boost_classifier.predict(X_test)
accuracy_train_set = grad_boost_classifier.score(X_train,y_train)
accuracy_test_set = grad_boost_classifier.score(X_test,y_test)
print(accuracy_train_set)
print(accuracy_test_set)

#%%

predictors = [x for x in X_train.columns]
param_test1 = {'n_estimators':range(20,81,10)}
gsearch1 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, min_samples_split=500,min_samples_leaf=50,max_depth=8,max_features='sqrt',subsample=0.8,random_state=10), 
param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch1.fit(train[predictors],train[y_train])

#%%

# confusion matrix for above trained model
confusion_matrix(y_test,predicted_values)
cm2 = confusion_matrix(y_test,predicted_values)
fig,ax = plt.subplots(figsize=(6,6))
ax.imshow(cm2)
plt.show()
# plotting heatmap for the confusion matrix
plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual values values');
plt.xlabel('Predicted values');
title1 = 'Score for testing dataset: {0}'.format(accuracy_test_set)
plt.title(title1, size = 15)

#%%

# segment a calcluation 
train_a = training_set.copy()
for i in train_a['label']:
    if train_a['label'].any() == 0 or 2 or 3 or 5 or 6 or 7 or 8 or 9:
        train_a['train_a'] = 1
    else :
        train_a['train_a'] = 0
print(train_a)


test_a = testing_set.copy()
for i in test_a.label:
    if test_a['label'].any() == 0 or 2 or 3 or 5 or 6 or 7 or 8 or 9:
        test_a['test_a'] = 1
    else :
        test_a = 0
    
y_train_a = train_a['label']
X_train_a = train_a.drop('label',axis=1)
X_train_a.drop('train_a',axis=1)
y_test_a = test_a['label']
X_test_a = test_a.drop('label',axis=1)
X_test_a.drop('test_a',axis=1)
y_train_a= y_train_a.values.reshape(-1,1)
y_test_a= y_test_a.values.reshape(-1,1)
scalar = StandardScaler()
scalar.fit(y_train_a)
scalar.fit(y_test_a)
 #print(train_a)    
y_train_a = train_a['label']
X_train_a = train_a.drop('label',axis=1)
y_test_a = test_a['label']
X_test_a = test_a.drop('label',axis=1)
#X_test=X_test
y_train_a= y_train_a.values.reshape(-1,1)
y_test_a= y_test_a.values.reshape(-1,1)
scalar = StandardScaler()
scalar.fit(y_train_a)
scalar.fit(y_test_a)

#%%

# Model training for learning rate 0.1 for segment a
grad_boost_classifier_a = GradientBoostingClassifier(learning_rate=0.1,max_depth=2,random_state=0)
grad_boost_classifier_a.fit(X_train_a,y_train_a)
predicted_values_a = grad_boost_classifier_a.predict(X_test_a)
accuracy_train_set_a = grad_boost_classifier_a.score(X_train_a,y_train_a)
accuracy_test_set_a = grad_boost_classifier_a.score(X_test_a,y_test_a)
print(accuracy_train_set_a)
print(accuracy_test_set_a)

#%%

# confusion matrix for above trained model for segment a 
confusion_matrix(y_test_a,predicted_values_a)
cm = confusion_matrix(y_test_a,predicted_values_a)
fig,ax = plt.subplots(figsize=(6,6))
ax.imshow(cm)
plt.show()
# plotting heatmap for the confusion matrix
plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual values values');
plt.xlabel('Predicted values');
title1 = 'Score for testing dataset: {0}'.format(accuracy_test_set_a)
plt.title(title1, size = 15)

#%%

train_b = training_set.copy()
for i in train_b['label']:
    if train_b['label'].any() == 0 or 1 or 2 or 3 or 4 or 7 or 8 or 9:
        train_b['train_b'] = 1
    else :
        train_b['train_b'] = 0
print(train_b)


test_b = testing_set.copy()
for i in test_b.label:
    if test_b['label'].any() == 0 or 1 or 2 or 3 or 4 or 7 or 8 or 9:
        test_b['test_a'] = 1
    else :
        test_b = 0 
 #print(train_b)    
y_train_b = train_b['label']
X_train_b = train_b.drop('label',axis=1)
#X_train_b.drop('train_b',axis=1)
y_test_b = test_b['label']
X_test_b = test_b.drop('label',axis=1)
#X_test_b.drop('test_b',axis =1)
#X_test=X_test
y_train_b= y_train_b.values.reshape(-1,1)
y_test_b= y_test_b.values.reshape(-1,1)
scalar = StandardScaler()
scalar.fit(y_train_b)
scalar.fit(y_test_b)

#%%

# Model training for learning rate 0.1 for segment b
grad_boost_classifier_b = GradientBoostingClassifier(learning_rate=0.1,max_depth=2,random_state=0)
grad_boost_classifier_b.fit(X_train_b,y_train_b)
predicted_values_b = grad_boost_classifier_b.predict(X_test_b)
accuracy_train_set_b = grad_boost_classifier_b.score(X_train_b,y_train_b)
accuracy_test_set_b = grad_boost_classifier_b.score(X_test_b,y_test_b)
print(accuracy_train_set_b)
print(accuracy_test_set_b)

#%%

# confusion matrix for above trained model for segment b
confusion_matrix(y_test_b,predicted_values_b)
cm = confusion_matrix(y_test_b,predicted_values_b)
fig,ax = plt.subplots(figsize=(6,6))
ax.imshow(cm)
plt.show()
# plotting heatmap for the confusion matrix
plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual values');
plt.xlabel('Predicted values');
title1 = 'Score for testing dataset: {0}'.format(accuracy_test_set_b)
plt.title(title1, size = 15)

#%%

train_c = training_set.copy()
for i in train_c['label']:
    if train_c['label'].any() == 0 or 1 or 3 or 4 or 5 or 6 or 7 or 8 or 9:
        train_c['train_c'] = 1
    else :
        train_c['train_c'] = 0
print(train_c)

 

test_c = testing_set.copy()
for i in test_c.label:
    if test_c['label'].any() == 0 or 1 or 3 or 4 or 5 or 6 or 7 or 8 or 9:
        test_c['test_c'] = 1
    else :
        test_c = 0 
 
 #print(train_c)    
y_train_c = train_c['label']
X_train_c = train_c.drop('label',axis=1)
X_train_c = X_train_c.drop('train_c',axis=1)
y_test_c = test_c['label']
X_test_c = test_c.drop('label',axis=1)
X_test_c = X_test_c.drop('test_c',axis=1)
#X_test=X_test
y_train_c = y_train_c.values.reshape(-1,1)
y_test_c = y_test_c.values.reshape(-1,1)
scalar = StandardScaler()
scalar.fit(y_train_c)
scalar.fit(y_test_c)

#%%

# Model training for learning rate 0.1 for segment c
grad_boost_classifier_c = GradientBoostingClassifier(learning_rate=0.1,max_depth=2,random_state=0)
grad_boost_classifier_c.fit(X_train_c,y_train_c)
predicted_values_c = grad_boost_classifier_c.predict(X_test_c)
accuracy_train_set_c = grad_boost_classifier_c.score(X_train_c,y_train_c)
accuracy_test_set_c = grad_boost_classifier_c.score(X_test_c,y_test_c)
print(accuracy_train_set_c)
print(accuracy_test_set_c)

#%%

# confusion matrix for above trained model for segment c
confusion_matrix(y_test_c,predicted_values_c)
cm = confusion_matrix(y_test_c,predicted_values_c)
fig,ax = plt.subplots(figsize=(6,6))
ax.imshow(cm)
plt.show()
# plotting heatmap for the confusion matrix
plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual values');
plt.xlabel('Predicted values');
title1 = 'Score for testing dataset: {0}'.format(accuracy_test_set_c)
plt.title(title1, size = 15)

#%%

train_d = training_set.copy()
for i in train_d['label']:
    if train_d['label'].any() == 0 or 2 or 3 or 5 or 6 or 8:
        train_d['train_d'] = 1
    else :
        train_d['train_d'] = 0
print(train_d)

test_d = testing_set.copy()
for i in test_d.label:
    if test_d['label'].any() == 0 or 2 or 3 or 5 or 6 or 8:
        test_d['test_d'] = 1
    else :
        test_d = 0 
        
 #print(train_d)    
y_train_d = train_d['label']
X_train_d = train_d.drop('label',axis=1)
X_train_d = X_train_d.drop('train_d',axis=1)
y_test_d = test_d['label']
X_test_d = test_d.drop('label',axis=1)
X_test_d = X_test_d.drop('test_d',axis=1)
#X_test=X_test
y_train_d = y_train_d.values.reshape(-1,1)
y_test_d = y_test_d.values.reshape(-1,1)
scalar = StandardScaler()
scalar.fit(y_train_d)
scalar.fit(y_test_d)

#%%

# Model training for learning rate 0.1 for segment d
grad_boost_classifier_d = GradientBoostingClassifier(learning_rate=0.1,max_depth=2,random_state=0)
grad_boost_classifier_d.fit(X_train_d,y_train_d)
predicted_values_d = grad_boost_classifier_d.predict(X_test_d)
accuracy_train_set_d = grad_boost_classifier_d.score(X_train_d,y_train_d)
accuracy_test_set_d = grad_boost_classifier_d.score(X_test_d,y_test_d)
print(accuracy_train_set_d)
print(accuracy_test_set_d)

#%%

# confusion matrix for above trained model for segment d
confusion_matrix(y_test_d,predicted_values_d)
cm = confusion_matrix(y_test_d,predicted_values_d)
fig,ax = plt.subplots(figsize=(6,6))
ax.imshow(cm)
plt.show()
# plotting heatmap for the confusion matrix
plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual values');
plt.xlabel('Predicted values');
title1 = 'Score for testing dataset: {0}'.format(accuracy_test_set_d)
plt.title(title1, size = 15)

#%%

train_e = training_set.copy()
for i in train_e['label']:
    if train_e['label'].any() == 0 or 2 or 6 or 8:
        train_e['train_e'] = 1
    else :
        train_e['train_e'] = 0
print(train_e)

test_e = testing_set.copy()
for i in test_e.label:
    if test_e['label'].any() == 0 or 2 or 6 or 8:
        test_e['test_e'] = 1
    else :
        test_e = 0 
 
 #print(train_e)    
y_train_e = train_e['label']
X_train_e = train_e.drop('label',axis=1)
y_test_e = test_e['label']
X_test_e = test_e.drop('label',axis=1)
#X_test=X_test
y_train_e = y_train_e.values.reshape(-1,1)
y_test_e = y_test_e.values.reshape(-1,1)
scalar = StandardScaler()
scalar.fit(y_train_e)
scalar.fit(y_test_e)

#%%

# Model training for learning rate 0.1 for segment e
grad_boost_classifier_e = GradientBoostingClassifier(learning_rate=0.1,max_depth=2,random_state=0)
grad_boost_classifier_e.fit(X_train_e,y_train_e)
predicted_values_e = grad_boost_classifier_e.predict(X_test_e)
accuracy_train_set_e = grad_boost_classifier_e.score(X_train_e,y_train_e)
accuracy_test_set_e = grad_boost_classifier_e.score(X_test_e,y_test_e)
print(accuracy_train_set_e)
print(accuracy_test_set_e)

#%%

# confusion matrix for above trained model for segment e
confusion_matrix(y_test_e,predicted_values_e)
cm = confusion_matrix(y_test_e,predicted_values_e)
fig,ax = plt.subplots(figsize=(6,6))
ax.imshow(cm)
plt.show()
# plotting heatmap for the confusion matrix
plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual values');
plt.xlabel('Predicted values');
title1 = 'Score for testing dataset: {0}'.format(accuracy_test_set_e)
plt.title(title1, size = 15)

#%%

train_f = training_set.copy()
for i in train_f['label']:
    if train_f['label'].any() == 0 or 4 or 5 or 6 or 8 or 9:
        train_f['train_f'] = 1
    else :
        train_f['train_f'] = 0
print(train_f)

test_f = testing_set.copy()
for i in test_f.label:
    if test_f['label'].any() == 0 or 4 or 5 or 6 or 8 or 9:
        test_f['test_f'] = 1
    else :
        test_f = 0 
 #print(train_f)    
y_train_f = train_f['label']
X_train_f = train_f.drop('label',axis=1)
y_test_f = test_f['label']
X_test_f = test_f.drop('label',axis=1)
#X_test=X_test
y_train_f = y_train_f.values.reshape(-1,1)
y_test_f = y_test_f.values.reshape(-1,1)
scalar = StandardScaler()
scalar.fit(y_train_f)
scalar.fit(y_test_f)

#%%

# Model training for learning rate 0.1 for segment f
grad_boost_classifier_f = GradientBoostingClassifier(learning_rate=0.1,max_depth=2,random_state=0)
grad_boost_classifier_f.fit(X_train_f,y_train_f)
predicted_values_f = grad_boost_classifier_f.predict(X_test_f)
accuracy_train_set_f = grad_boost_classifier_f.score(X_train_f,y_train_f)
accuracy_test_set_f = grad_boost_classifier_f.score(X_test_f,y_test_f)
print(accuracy_train_set_f)
print(accuracy_test_set_f)

#%%

# confusion matrix for above trained model for segment f
confusion_matrix(y_test_f,predicted_values_f)
cm = confusion_matrix(y_test_f,predicted_values_f)
fig,ax = plt.subplots(figsize=(6,6))
ax.imshow(cm)
plt.show()
# plotting heatmap for the confusion matrix
plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual values');
plt.xlabel('Predicted values');
title1 = 'Score for testing dataset: {0}'.format(accuracy_test_set_f)
plt.title(title1, size = 15)

#%%

train_g = training_set.copy()
for i in train_g['label']:
    if train_g['label'].any() == 2 or 3 or 4 or 5 or 6 or 8 or 9:
        train_g['train_g'] = 1
    else :
        train_g['train_g'] = 0
print(train_g)

test_g = testing_set.copy()
for i in test_g.label:
    if test_g['label'].any() == 2 or 3 or 4 or 5 or 6 or 8 or 9:
        test_g['test_g'] = 1
    else :
        test_g = 0 

 
 #print(train_g)    
y_train_g = train_g['label']
X_train_g = train_g.drop('label',axis=1)
y_test_g = test_g['label']
X_test_g = test_g.drop('label',axis=1)
#X_test=X_test
y_train_g = y_train_g.values.reshape(-1,1)
y_test_g = y_test_g.values.reshape(-1,1)
scalar = StandardScaler()
scalar.fit(y_train_g)
scalar.fit(y_test_g)

#%%

# Model training for learning rate 0.1 for segment g
grad_boost_classifier_g = GradientBoostingClassifier(learning_rate=0.1,max_depth=2,random_state=0)
grad_boost_classifier_g.fit(X_train_g,y_train_g)
predicted_values_g = grad_boost_classifier_g.predict(X_test_g)
accuracy_train_set_g = grad_boost_classifier_g.score(X_train_g,y_train_g)
accuracy_test_set_g = grad_boost_classifier_g.score(X_test_g,y_test_g)
print(accuracy_train_set_g)
print(accuracy_test_set_g)

#%%

# confusion matrix for above trained model for segment g
confusion_matrix(y_test_g,predicted_values_g)
cm = confusion_matrix(y_test_g,predicted_values_g)
fig,ax = plt.subplots(figsize=(6,6))
ax.imshow(cm)
plt.show()
# plotting heatmap for the confusion matrix
plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual values');
plt.xlabel('Predicted values');
title1 = 'Score for testing dataset: {0}'.format(accuracy_test_set_g)
plt.title(title1, size = 15)
