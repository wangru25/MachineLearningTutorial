import numpy as np
import pandas as pd
import math
import time
import random
from nn.activations import sigmoid_forward, sigmoid_backward, tanh_forward, tanh_backward
np.random.seed(7)

tic  = time.clock()

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


# def one_hot_encoder(y_train, y_test):
#     ''' convert label to a vector under one-hot-code fashion '''
#     from sklearn import preprocessing
#     lb = preprocessing.LabelBinarizer()
#     lb.fit(y_train)
#     y_train_ohe = lb.transform(y_train)
#     y_test_ohe = lb.transform(y_test)
#     return y_train_ohe, y_test_ohe


X_train, y_train = read_dataset('airfoil_self_noise_X_train.csv', 'airfoil_self_noise_y_train.csv')
X_test, y_test = read_dataset('airfoil_self_noise_X_test.csv', 'airfoil_self_noise_y_test.csv')
X_train_norm, X_test_norm = normalize_features(X_train, X_test)

print(X_train_norm.shape)
print(X_test_norm.shape)
print(y_train.shape)
print(y_test.shape)


class ANN_Regr:
    def __init__(self, X, y, activation = 'relu', hidden_layer_nn_1=100, hidden_layer_nn_2=100, lr=0.01,):
        if activation == 'relu':
            self.activation_forward = relu_forward
            self.activation_backward = relu_backward
        elif activation == 'tanh':
            self.activation_forward = tanh_forward
            self.activation_backward = tanh_backward
        elif activation == 'sigmoid':
            self.activation_forward = sigmoid_forward
            self.activation_backward = sigmoid_backward

        self.X = X
        self.y = y
        self.hidden_layer_nn_1 = hidden_layer_nn_1
        self.hidden_layer_nn_2 = hidden_layer_nn_2
        self.lr = lr
        self.nn = X.shape[1]
        self.W1 = np.random.randn(self.nn, hidden_layer_nn_1) / self.nn
        self.b1 = np.zeros((1, hidden_layer_nn_1))
        self.W2 = np.random.randn(hidden_layer_nn_1, self.hidden_layer_nn_2) / self.hidden_layer_nn_1
        self.b2 = np.zeros((1, self.hidden_layer_nn_2))
        self.W3 = np.random.randn(hidden_layer_nn_2, 1) / self.hidden_layer_nn_2
        self.b3 = 0.0

    def feed_forward(self):
        self.z1 = np.dot(self.X, self.W1) + self.b1
        self.f1 = self.activation_forward(self.z1)
        self.z2 = np.dot(self.f1, self.W2) + self.b2
        self.f2 = self.activation_forward(self.z2)
        self.y_hat = np.dot(self.f2, self.W3) + self.b3

    def back_propagation(self):
        d3 = (self.y_hat - self.y) / self.nn
        dW3 = np.dot(self.f2.T, d3)
        db3 = np.sum(d3, axis=0, keepdims=True)
        d2 = self.activation_backward(self.z2) * (d3.dot((self.W3).T))
        dW2 = np.dot(self.f1.T, d2)
        db2 = np.sum(d2, axis=0, keepdims=True)
        d1 = self.activation_backward(self.z1) * (d2.dot((self.W2).T))
        dW1 = np.dot(self.X.T, d1)
        db1 = np.sum(d1, axis=0, keepdims=True)

        # Update the gradident descent
        self.W1 = self.W1 - self.lr * dW1
        self.b1 = self.b1 - self.lr * db1
        self.W2 = self.W2 - self.lr * dW2
        self.b2 = self.b2 - self.lr * db2
        self.W3 = self.W3 - self.lr * dW3
        self.b3 = self.b3 - self.lr * db3

    def lossfunction(self):

        self.feed_forward()
        # self.loss = 0.5 * np.sum((self.y_hat - self.y)**2) / self.X.shape[0] + (0.5*self.Lambda)*(np.sum(self.W1**2)+np.sum(self.W2**2)+np.sum(self.W3**2))
        self.loss = 0.5 * np.sum((self.y_hat - self.y)**2) / self.X.shape[0]

    def predict(self, X_test):
        z1 = np.dot(X_test, self.W1) + self.b1
        f1 = self.activation_forward(z1)
        z2 = np.dot(f1, self.W2) + self.b2
        f2 = self.activation_forward(z2)
        y_hat_test = np.dot(f2, self.W3) + self.b3
        return y_hat_test

def RMSE(ypred, yexact):
    return np.sqrt(np.sum((ypred - yexact)**2)/ ypred.shape[0])

def relu_forward(z):
    '''This is the finction for ReLU'''
    return z * (z > 0)

def relu_backward(z):
    '''This is the derivative of ReLU'''
    return 1. * (z > 0)

myANN_Regr = ANN_Regr(X_train_norm, y_train, activation = 'sigmoid', hidden_layer_nn_1 = 5, hidden_layer_nn_2 = 3, lr=0.0001)
epoch_num = 200
for i in range(epoch_num):
    myANN_Regr.feed_forward()
    myANN_Regr.back_propagation()
    myANN_Regr.lossfunction()
    if ((i+1)% 50 == 0):
        print('epoch = %d, current loss = %.5f' % (i+1, myANN_Regr.loss))

y_pred = myANN_Regr.predict(X_test_norm)
print(y_pred.ravel())
print(y_test.ravel())
print('RMSE of our model ', RMSE(y_pred, y_test))

toc = time.clock()

print('Totol time:' + str((toc-tic))+ 's')

print('===============================Finish===================================')
