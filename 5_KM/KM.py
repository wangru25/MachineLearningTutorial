# -*- coding: utf-8 -*-
'''
Author: Rui Wang
Date: 2019-01-17 11:48:43
LastModifiedBy: Rui Wang
LastEditTime: 2022-04-10 14:38:50
Email: wangru25@msu.edu
FilePath: /5_KM/KM.py
Description: 
'''
import math
import random
from re import X
import numpy as np
import pandas as pd
from scipy.spatial import distance
from scipy import stats
# import time

random.seed(1)


# tic = time.clock()


def read_dataset(feature_file, label_file):
    ''' Read data set in *.csv to data frame in Pandas'''
    df_X = pd.read_csv(feature_file)
    df_y = pd.read_csv(label_file)
    X = df_X.values  # convert values in dataframe to numpy array (features)
    y = df_y.values  # convert values in dataframe to numpy array (label)
    return X, y

def normalize_features(X_train, X_test):
    from sklearn.preprocessing import StandardScaler, MinMaxScaler  # import libaray
    scaler = StandardScaler()  # call an object function
    scaler.fit(X_train)  # calculate mean, std in X_train
    X_train_norm = scaler.transform(X_train)  # apply normalization on X_train
    # we use the same normalization on X_test
    X_test_norm = scaler.transform(X_test)
    return X_train_norm, X_test_norm


X_train, y_train = read_dataset('MNIST_X_train.csv', 'MNIST_y_train.csv')
X_test, y_test = read_dataset('MNIST_X_test.csv', 'MNIST_y_test.csv')
X_train_norm, X_test_norm = normalize_features(X_train, X_test)


print(X_train_norm.shape)
print(X_test_norm.shape)
print(y_train.shape)
print(y_test.shape)


class KMeans:
    def __init__(self, X, k=5, iters=1000):
        self.X = X
        self.k = k
        self.iters = iters
        self.m = self.X.shape[0]  # numbers of samples
        self.n = self.X.shape[1]  # numbers of fearures
        self.centers = np.random.randn(self.k, self.n)
        self.smallest_dist_idx = None

    def fit(self):
        for _ in range(self.iters):
            self.update_center()

    def update_center(self):
        dist_to_center = distance.cdist(self.X, self.centers)
        self.smallest_dist_idx = dist_to_center.argmin(axis=1)
        for i in range(self.k):
            self.centers[i] = np.mean(self.X[self.smallest_dist_idx == i], axis=0)

    def WCSS(self):
        '''
        Within Cluster Sum of Squares
        - indicate the variablily of the points within the cluster
        in terms of the average distance to the cluster center
        '''
        score = 0.0
        for i in range(self.k):
            score += np.sum((np.sum(self.X[self.smallest_dist_idx == i] - self.centers[i])**2),axis=0)
        return score


myKM = KMeans(X_train_norm, k=3, iters=1000)
myKM.fit()
# myKM.update_center()
wcss = myKM.WCSS()
print('WCSS of our model', wcss)


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5, random_state=0).fit(X_test_norm)
kmeans.labels_

X1 = kmeans.predict(X_test_norm)

centers = kmeans.cluster_centers_
print(X1.shape)
print(kmeans.inertia_)

print(centers.shape)
print()


# toc = time.clock()

# print('Totol time:' + str((toc - tic)) + 's')

print('===============================Finish===================================')



