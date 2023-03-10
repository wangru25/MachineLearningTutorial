import numpy as np
import pandas as pd
import random

# np.random.seed(1)

def readcsv(feature, label):  #read csv
    x = pd.read_csv(feature)
    y = pd.read_csv(label)
    x_values = x.values
    y_values = y.values
    return x_values, y_values


def norm(x_train, x_test):  # norm values
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train_norm = scaler.transform(x_train)
    x_test_norm = scaler.transform(x_test)
    return x_train_norm, x_test_norm


class SVM():
    def __init__(self, x, y, h=0.01, alpha=0.01):
        self.x = x
        self.y = y
        self.h = h
        self.m1, self.n1 = x.shape
        self.m2, self.n2 = y.shape
        self.alpha = alpha
#         self.c = np.ones((self.n1, self.n2))
        self.c = np.random.randn(self.n1, self.n2)
        self.b = np.zeros((1, self.n2))
        print(x.shape)
        print(y.shape)

    def forward(self):
        self.y_hat = np.dot(self.x, self.c) + self.b
        self.cond = 1 - self.y * self.y_hat

    def gd(self):
#         self.c = self.c - self.h * (self.c / np.linalg.norm(self.c, axis=0))
        # y_hat = self.y.copy()
        y_hat = np.where(self.cond > 0, self.y, 0)
        dc = (1/np.linalg.norm(self.c, axis=0)) * self.c - self.alpha * np.dot(y_hat.T, self.x).T
        db = self.alpha * np.sum(-y_hat ,axis = 0)
        self.c = self.c - self.h * dc
        self.b = self.b - self.h * db
        # self.c = self.c - self.h * (1/ np.linalg.norm(self.c, axis=0)) * self.c + self.h * self.alpha * (np.dot(y_hat.T, self.x).T)
        # self.b = self.b + self.h * self.alpha * np.sum(y_hat,axis=0)

    def predict(self, x_test):
        m = x_test.shape[0]
#         return np.dot(x_test, self.c) + np.tile(self.b, (m, 1))
        return np.dot(x_test, self.c) + self.b



def one_hot_encoder(y_train, y_test):
    ''' convert label to a vector under one-hot-code fashion '''
    from sklearn import preprocessing
    lb = preprocessing.LabelBinarizer()
    lb.fit(y_train)
    y_train_ohe = lb.transform(y_train)
    y_test_ohe = lb.transform(y_test)
    return y_train_ohe, y_test_ohe

# def one_hot_encoder(y_train, y_test):
#     ''' convert label to a vector under one-hot-code fashion '''
#     from sklearn import preprocessing
#     lb = preprocessing.LabelBinarizer()
#     lb.fit(y_train)
#     y_train_ohe1 = lb.transform(y_train)
#     y_test_ohe1 = lb.transform(y_test)
#     y_train_ohe = np.where(y_train_ohe1 > 0, y_train_ohe1, -1)
#     y_test_ohe = np.where(y_test_ohe1 > 0, y_test_ohe1, -1)
#     return y_train_ohe, y_test_ohe


def one_hot_decoder(y_test):
    y_test_real = y_test.argmax(axis=1)
    return y_test_real.reshape(-1, 1)


def accuracy(y_real, y_pre):
    z = y_pre - y_real
    error = np.sum(z == 0)
    m = y_pre.shape[0]
    # print(m)
    # print(error)
    return error / float(m)

x_train, y_train = readcsv('MNIST_X_train.csv', 'MNIST_y_train.csv')
x_test, y_test = readcsv('MNIST_X_test.csv', 'MNIST_y_test.csv')
x_train_norm, x_test_norm = norm(x_train, x_test)
y_train_ohe, y_test_ohe = one_hot_encoder(y_train, y_test)

SVM1 = SVM(x_train_norm, y_train_ohe, h=0.01, alpha=0.01)

epoch = 2000
for i in range(epoch):
    SVM1.forward()
    SVM1.gd()

y_test_ohe_hat = SVM1.predict(x_test_norm)
y_test_hat = one_hot_decoder(y_test_ohe_hat)
acc = accuracy(y_test, y_test_hat)
print(acc)
