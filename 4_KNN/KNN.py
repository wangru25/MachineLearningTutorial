# -*- coding: utf-8 -*-
'''
Author: Rui Wang
Date: 2020-08-26 16:48:10
LastModifiedBy: Rui Wang
LastEditTime: 2022-05-29 13:58:32
Email: wangru25@msu.edu
FilePath: /undefined/Users/rui/Dropbox/Linux_Backup/MSU/1_Training/4_KNN/KNN.py
Description: 
'''
import math
import numpy as np
import pandas as pd
from scipy.spatial import distance
from scipy import stats
import time

import seaborn as sns
sns.set()

import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

# tic = time.clock()

def join_dataset(feature_file, label_file):
    df_X = pd.read_csv(feature_file)
    df_y = pd.read_csv(label_file)
    iris = pd.concat([df_X, df_y], axis=1)
    return iris

def read_dataset(feature_file, label_file):
    ''' Read data set in *.csv to data frame in Pandas'''
    df_X = pd.read_csv(feature_file)
    df_y = pd.read_csv(label_file)
    X = df_X.values # convert values in dataframe to numpy array (features)
    y = df_y.values # convert values in dataframe to numpy array (label)
    return X, y


def normalize_features(X_train, X_test):
    from sklearn.preprocessing import StandardScaler #import libaray.  #MinMaxScaler
    scaler = StandardScaler() # call an object function
    scaler.fit(X_train) # calculate mean, std in X_train
    X_train_norm = scaler.transform(X_train) # apply normalization on X_train
    X_test_norm = scaler.transform(X_test) # we use the same normalization on X_test
    return X_train_norm, X_test_norm


# X_train, y_train = read_dataset('MNIST_X_train.csv', 'MNIST_y_train.csv')
# X_test, y_test = read_dataset('MNIST_X_test.csv', 'MNIST_y_test.csv')
# iris_train = join_dataset('MNIST_X_train.csv', 'MNIST_y_train.csv')
# iris_test = join_dataset('MNIST_X_test.csv', 'MNIST_y_test.csv') # mainly used for plotting
X_train, y_train = read_dataset('Iris_X_train.csv', 'Iris_y_train.csv')
X_test, y_test = read_dataset('Iris_X_test.csv', 'Iris_y_test.csv')
iris_train = join_dataset('Iris_X_train.csv', 'Iris_y_train.csv')
iris_test = join_dataset('Iris_X_test.csv', 'Iris_y_test.csv') # mainly used for plotting
X_train_norm, X_test_norm = normalize_features(X_train, X_test)


sns.scatterplot(x="sepal width (cm)", y="sepal length (cm)",hue='TYPE', data=iris_train)
plt.show()

print(X_train_norm.shape)
print(X_test_norm.shape)
print(y_train.shape)
print(y_test.shape)


class KNN:
    def __init__(self, X, y, k=5):
        self.X = X
        self.y = y
        self.k = k

    def predict(self, X_test):
        dist =  distance.cdist(self.X, X_test, 'euclidean')
        num_test_samples = X_test.shape[0]
        ypred = np.zeros(num_test_samples, dtype=int)
        for i in range(num_test_samples):
            closest_y = []
            k_nearest_idx = np.argsort(dist[:,i])[0 : self.k]
            closest_y = self.y[k_nearest_idx]
            ypred[i] = stats.mode(closest_y)[0]
        return ypred

    # def predict(self, X_test):
    #     dist =  distance.cdist(self.X, X_test, 'euclidean')
    #     num_test_samples = X_test.shape[0]
    #     ypred = np.zeros(num_test_samples, dtype=int)
    #     k_nearest_idx = np.argsort(dist, axis = 0)[0:self.k, :]
    #     closest_y = self.y[k_nearest_idx]
    #     ypred = stats.mode(closest_y)[0]
    #     return ypred

def Accuracy(ypred, yexact):
    p = np.array(ypred == yexact, dtype = int)
    return np.sum(p)/float(len(yexact))

myKNN = KNN(X_train_norm, y_train, k=3)
y_pred = myKNN.predict(X_test_norm)
print('Accuracy of our model', Accuracy(y_pred, y_test.ravel()))


# toc = time.clock()

# print('Totol time:' + str((toc-tic))+ 's')

print('===============================Finish===================================')


