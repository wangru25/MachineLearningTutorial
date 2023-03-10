import math
import time
import numpy as np
import pandas as pd
from nn.Layers import rnn_cell_forward, rnn_forward
from nn.Layers import rnn_cell_backward, rnn_backward
from nn.activations import tanh, sigmoid, softmax
tic = time.clock()

'''
In this file, we will use dictionary to combine all the seperate layers together
'''


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
    X_train_norm1 = scaler.transform(X_train) # apply normalization on X_train
    X_test_norm1 = scaler.transform(X_test) # we use the same normalization on X_test
    X_train_norm = np.reshape(X_train_norm1,(-1,28,28)) # reshape X to be a 4-D array
    X_test_norm = np.reshape(X_test_norm1,(-1,28,28))
    return X_train_norm, X_test_norm


def one_hot_encoder(y_train, y_test):
    ''' convert label to a vector under one-hot-code fashion '''
    from sklearn import preprocessing
    lb = preprocessing.LabelBinarizer()
    lb.fit(y_train)
    y_train_ohe = lb.transform(y_train)
    y_test_ohe = lb.transform(y_test)
    return y_train_ohe, y_test_ohe



X_train, y_train = read_dataset('MNIST_X_train.csv', 'MNIST_y_train.csv')
X_test, y_test = read_dataset('MNIST_X_test.csv', 'MNIST_y_test.csv')
X_train_norm, X_test_norm = normalize_features(X_train, X_test)
y_train_ohe, y_test_ohe = one_hot_encoder(y_train, y_test)

print(X_train_norm.shape)
print(X_test_norm.shape)
print(y_train_ohe.shape)
print(y_test_ohe.shape)

H = 128
N = y_test_ohe.shape[0]
# N = X_test_norm.shape[0]
D = X_test_norm.shape[1]
T = X_test_norm.shape[2]
M = y_test_ohe.shape[1]
h0 = np.zeros((N, H))

Wx = np.random.randn(D, H)
Wh = np.random.randn(H, H)
Wy = np.random.randn(H, M)
bh = np.zeros((1,H))
by = np.zeros((1,M))
h0 = np.zeros((N, H))
parameters = {"Wh": Wh, "Wx": Wx, "Wy": Wy, "bh": bh, "by": by}
h, y_hat, caches = rnn_forward(X_test_norm, h0, parameters)
print(y_hat[:,:,27].shape)
# print(caches[0][0])def predict(self, X_test):

def predict(y_pred,X_test):
    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    num_test_samples = X_test.shape[0]
    ypred = np.zeros(num_test_samples, dtype=int)
    for i in range(num_test_samples):
        ypred[i] = labels[np.argmax(y_pred[i,:])]
    return ypred

y_pred = y_hat[:,:,27]
ypred = predict(y_pred,X_test_norm)
print(ypred)
print(y_test.ravel())

def accuracy(ypred, yexact):
    # p = np.array(ypred == yexact, dtype = int)
    p = np.array(ypred == yexact, dtype = int)
    return np.sum(p)/float(len(yexact))

print('Accuracy of our model ', accuracy(y_pred, y_test.ravel()))

toc = time.clock()
print('Totol time:' + str((toc-tic))+ 's')
print('===============================Finish===================================')
