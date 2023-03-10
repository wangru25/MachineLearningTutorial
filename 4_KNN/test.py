# -*- coding: utf-8 -*-
'''
Author: Rui Wang
Date: 2019-11-19 16:21:39
LastModifiedBy: Rui Wang
LastEditTime: 2020-09-01 17:03:53
Email: wangru25@msu.edu
FilePath: /4_KNN/test.py
Description: 
'''
# from scipy.spatial import distance
# import numpy as np
# a = np.array([[0, 0, 0],
#               [0, 0, 1],
#               [0, 1, 0],
#               [0, 1, 1],
#               [1, 0, 0],
#               [1, 0, 1],
#               [1, 1, 0],
#               [1, 1, 1]])
# b = np.array([[ 0.1,  0.2,  0.4],
#               [0.1, 0.2, 0.2]])
# c = distance.cdist(a, b, 'cityblock')
# print(c)

# import numpy as np
# a = np.array([[6, 8, 3, 0],
#               [3, 2, 1, 0],
#               [8, 1, 8, 4],
#               [5, 3, 0, 5],
#               [4, 7, 5, 5]])
# from scipy import stats
# b = stats.mode(a)[0]
# print(b)

# import math
# import numpy as np
# import pandas as pd
# from scipy.spatial import distance
# from scipy import stats
# import time
#
# import seaborn as sns
# sns.set()
#
#
#
# tic = time.clock()
#
# def join_dataset(feature_file, label_file):
#     df_X = pd.read_csv(feature_file)
#     df_y = pd.read_csv(label_file)
#     iris = pd.concat([df_X, df_y], axis=1)
#     return iris
#
# def read_dataset(feature_file, label_file):
#     ''' Read data set in *.csv to data frame in Pandas'''
#     df_X = pd.read_csv(feature_file)
#     df_y = pd.read_csv(label_file)
#     X = df_X.values # convert values in dataframe to numpy array (features)
#     y = df_y.values # convert values in dataframe to numpy array (label)
#     return X, y
#
#
# def normalize_features(X_train, X_test):
#     from sklearn.preprocessing import StandardScaler #import libaray
#     scaler = StandardScaler() # call an object function
#     scaler.fit(X_train) # calculate mean, std in X_train
#     X_train_norm = scaler.transform(X_train) # apply normalization on X_train
#     X_test_norm = scaler.transform(X_test) # we use the same normalization on X_test
#     return X_train_norm, X_test_norm
#
# X_train, y_train = read_dataset('Iris_X_train.csv', 'Iris_y_train.csv')
# X_test, y_test = read_dataset('Iris_X_test.csv', 'Iris_y_test.csv')
# iris_train = join_dataset('Iris_X_train.csv', 'Iris_y_train.csv')
# iris_test = join_dataset('Iris_X_test.csv', 'Iris_y_test.csv') # mainly used for plotting
# X_train_norm, X_test_norm = normalize_features(X_train, X_test)
#
# # sns.scatterplot(x="sepal width (cm)", y="sepal length (cm)",hue='TYPE', data=iris_train)
# # plt.show()
#
# print(X_train_norm.shape)
# print(X_test_norm.shape)
# print(y_train.shape)
# print(y_test.shape)
#
# dist =  distance.cdist(X_train_norm, X_test_norm, 'euclidean')
# print(type(dist))
# print(dist.shape)
#
# toc = time.clock()
#
# print('Totol time:' + str((toc-tic))+ 's')
#
# print('===============================Finish===================================')

import math
import numpy as np
import pandas as pd
from scipy.spatial import distance
from scipy import stats
import time

def read_dataset(feature_file, label_file):
    ''' Read data set in *.csv to data frame in Pandas'''
    df_X = pd.read_csv(feature_file)
    df_y = pd.read_csv(label_file)
    X = df_X.values # convert values in dataframe to numpy array (features)
    y = df_y.values # convert values in dataframe to numpy array (label)
    return X, y


def normalize_features(X_train, X_test):
    from sklearn.preprocessing import StandardScaler #import libaray
    scaler = StandardScaler() # call an object function
    scaler.fit(X_train) # calculate mean, std in X_train
    X_train_norm = scaler.transform(X_train) # apply normalization on X_train
    X_test_norm = scaler.transform(X_test) # we use the same normalization on X_test
    return X_train_norm, X_test_norm

X_train, y_train = read_dataset('Iris_X_train.csv', 'Iris_y_train.csv')
X_test, y_test = read_dataset('Iris_X_test.csv', 'Iris_y_test.csv')
# X_train_norm, X_test_norm = normalize_features(X_train, X_test)




from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=4, algorithm='brute')
neigh.fit(X_train, y_train)
y_pred = neigh.predict(X_test)
print(y_pred)

y_1 = neigh.predict(X_train)


def Accuracy(ypred, yexact):
    p = np.array(ypred == yexact, dtype = int)
    return np.sum(p)/float(len(yexact))

print('Accuracy of our model', Accuracy(y_pred.ravel(), y_test.ravel()))
print('Accuracy of our model', Accuracy(y_1.ravel(), y_train.ravel()))
