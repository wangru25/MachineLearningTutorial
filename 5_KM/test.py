import numpy as np
import pandas as pd
import random
from scipy.spatial import distance
from scipy import stats

def read_dataset(feature_file, label_file):
    ''' Read data set in *.csv to data frame in Pandas'''
    df_X = pd.read_csv(feature_file)
    df_y = pd.read_csv(label_file)
    X = df_X.values  # convert values in dataframe to numpy array (features)
    y = df_y.values  # convert values in dataframe to numpy array (label)
    return X, y


def normalize_features(X_train, X_test):
    from sklearn.preprocessing import StandardScaler  # import libaray
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

centers = np.random.randn(5, X_test.shape[1])
dist_to_center = distance.cdist(X_test, centers)
smallest_dist_idx = dist_to_center.argmin(axis=1)
a = X_test[smallest_dist_idx == 1]
# for i in range(5):
#     centers[i] = np.mean(X_test[smallest_dist_idx == i], axis = 0)
# print(centers)
# print(centers.shape)
# print(smallest_dist_idx)
# print(smallest_dist_idx.shape)
print(a)
print(a.shape)
