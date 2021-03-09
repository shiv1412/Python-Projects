# -*- coding: utf-8 -*-
"""
Created on Sat May  9 20:46:54 2020

@author:  Shivani Sharma
"""


#import libraries
import pandas as pd
import numpy as np
import random as rd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#implementation using pandas ,numpy and matplotlib
data = pd.read_csv('D:/Big Data Management/201912-bluebikes-tripdata.csv')
data.head()

X = data[["tripduration","birth.year"]]
#Visualise data points
plt.scatter(X["birth.year"],X["tripduration"],c='black')
plt.xlabel('birth.year')
plt.ylabel('tripduration')
plt.show()


# Step 1 and 2 - Choose the number of clusters (k) and select random centroid
# for each cluster number of clusters
K=2
# Select random observation as centroids
Centroids = (X.sample(n=K))
plt.scatter(X["birth.year"],X["tripduration"],c='black')
plt.scatter(Centroids["tripduration"],Centroids["birth.year"],c='red')
plt.xlabel('birth.year')
plt.ylabel('tripduration')
plt.show()


# algorithm logic from scratch 
# Step 3 - Assign all the points to the closest cluster centroid
# Step 4 - Recompute centroids of newly formed clusters
# Step 5 - Repeat step 3 and 4

diff = 1
j=0

while(diff!=0):
    XD=X
    i=1
    for index1,row_c in Centroids.iterrows():
        ED=[]
        for index2,row_d in XD.iterrows():
            d1=(row_c["birth.year"]-row_d["birth.year"])**2
            d2=(row_c["tripduration"]-row_d["tripduration"])**2
            d=np.sqrt(d1+d2)
            ED.append(d)
        X[i]=ED
        i=i+1

    C=[]
    for index,row in X.iterrows():
        min_dist=row[1]
        pos=1
        for i in range(K):
            if row[i+1] < min_dist:
                min_dist = row[i+1]
                pos=i+1
        C.append(pos)
    X["Cluster"]=C
    Centroids_new = X.groupby(["Cluster"]).mean()[["tripduration","birth.year"]]
    if j == 0:
        diff=1
        j=j+1
    else:
        diff = (Centroids_new['tripduration'] - Centroids['tripduration']).sum() + (Centroids_new['birth.year'] - Centroids['birth.year']).sum()
        print(diff.sum())
    Centroids = X.groupby(["Cluster"]).mean()[["tripduration","birth.year"]]
    

color=['blue','green','cyan']
for k in range(K):
    data=X[X["Cluster"]==k+1]
    plt.scatter(data["birth.year"],data["tripduration"],c=color[k])
plt.scatter(Centroids["birth.year"],Centroids["tripduration"],c='red')
plt.xlabel('birth.year')
plt.ylabel('tripduration')
plt.show()



#Implementation using sklean library

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
# statistics of scaled data
pd.DataFrame(data_scaled).describe()
# defining the kmeans function with initialization as k-means++
kmeans = KMeans(n_clusters=2, init='k-means++')

# fitting the k means algorithm on scaled data
kmeans.fit(data_scaled)
# inertia on the fitted data
kmeans.inertia_

# fitting multiple k-means algorithms and storing the values in an empty list
SSE = []
for cluster in range(1,18):
    kmeans = KMeans(n_jobs = -1, n_clusters = cluster, init='k-means++')
    kmeans.fit(data_scaled)
    SSE.append(kmeans.inertia_)
# converting the results into a dataframe and plotting them
frame = pd.DataFrame({'Cluster':range(1,18), 'SSE':SSE})
plt.figure(figsize=(12,6))
plt.plot(frame['Cluster'], frame['SSE'], marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')

# k means using 5 clusters and k-means++ initialization
kmeans = KMeans(n_jobs = -1, n_clusters = 5, init='k-means++')
kmeans.fit(data_scaled)
pred = kmeans.predict(data_scaled)
plt.show()
frame = pd.DataFrame(data_scaled)
frame['cluster'] = pred
frame['cluster'].value_counts()
