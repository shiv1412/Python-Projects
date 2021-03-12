# Importing all the required libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split as split
import seaborn as sn

# invoking required methods in main function
def main():
    knn_model()

def knn_model():
    # data set reading
    train_dataset = pd.read_csv('D:/Predictive Course/fashion/fashion_train.csv')
    test_dataset = pd.read_csv('D:/Predictive Course/fashion/fashion_test.csv')
    label = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
 
    # splitting the datasets into train and test parts for labels and features 
    y_train = train_dataset['label']
    y_test = test_dataset['label']
    X_train = train_dataset.drop('label',axis=1)
    X_test = test_dataset.drop('label',axis=1)
   
    #Applying knn classifier for k=1 value and calculating accuracy and finding confusion matrix
    knn_model1 = KNeighborsClassifier(n_neighbors=1)
    fit1=knn_model1.fit(X_train, y_train)
    knn_1_prediction = knn_model1.predict(X_test)
    accuracy_k1 = metrics.accuracy_score(y_test, knn_1_prediction)
    print(accuracy_k1)
    conf_met1 = confusion_matrix(y_test, knn_1_prediction)
    print(conf_met1)
    matrix1 = plot_confusion_matrix(fit1, X_test, y_test,
                                 cmap=plt.cm.Blues,
                                 normalize='true')
    plt.title('Confusion matrix for our classifier')
    plt.show(matrix1)
    #plt.show()   
    #Applying knn classifier for k=11 value and calculating accuracy and finding confusion matri
    knn_model2 = KNeighborsClassifier(n_neighbors=11)
    fit2=knn_model2.fit(X_train, y_train)
    knn_2_prediction = knn_model2.predict(X_test)
    accuracy_k2 = metrics.accuracy_score(y_test, knn_2_prediction)
    print(accuracy_k2)
    conf_met2 = confusion_matrix(y_test, knn_2_prediction)
    print(conf_met2)
    # plt.show(conf_met2)
    matrix2 = plot_confusion_matrix(fit2, X_test, y_test,
                                 cmap=plt.cm.Blues,
                                 normalize='true')
    plt.title('Confusion matrix for our classifier')
    plt.show(matrix2)
    #plt.show()
    
    #Applying knn classifier for k=21 value and calculating accuracy and finding confusion matrix
    knn_model3 = KNeighborsClassifier(n_neighbors=21)
    fit3=knn_model3.fit(X_train, y_train)
    knn_3_prediction = knn_model3.predict(X_test)
    accuracy_k3 = metrics.accuracy_score(y_test, knn_3_prediction)
    print(accuracy_k3)
    conf_met3 = confusion_matrix(y_test, knn_3_prediction)
    print(conf_met3)
    # plt.show(conf_met3)
    # plotting confusion matrix
    title3 = [("Confusion matrix for k=21", None),
                  ("Normalized value of confusion matrix", 'true')]
    matrix3 = plot_confusion_matrix(fit3, X_test, y_test,
                                 cmap=plt.cm.Blues,
                                 normalize='true')
    plt.title('Confusion matrix for our classifier')
    plt.show(matrix3)
    #plt.show()
    

if __name__ == '__main__':
    main()

