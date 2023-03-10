import math
import numpy as np
import pandas as pd
import time

tic = time.clock()

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

X_train, y_train = read_dataset('airfoil_self_noise_X_train.csv', 'airfoil_self_noise_y_train.csv')
X_test, y_test = read_dataset('airfoil_self_noise_X_test.csv', 'airfoil_self_noise_y_test.csv')
X_train_norm_org, X_test_norm_org = normalize_features(X_train, X_test)

# Add one column with all elements to be 1
X_train_norm = np.hstack((np.ones((y_train.shape[0],1)), X_train_norm_org))
X_test_norm = np.hstack((np.ones((y_test.shape[0],1)), X_test_norm_org))

print(X_train_norm.shape)
print(X_test_norm.shape)
print(y_train.shape)
print(y_test.shape)

class LinearR:
    def __init__(self, X, y, lr = 0.01):
        self.X = X
        self.y = y
        self.lr = lr
        self.m = X.shape[1]
        self.W = np.zeros((self.m,1))

    def forward(self):
        self.y_hat = np.dot(self.X, self.W)

    def gradientDescent(self):
        d = (self.y_hat - self.y) / self.X.shape[0]
        dW = np.dot(self.X.T, d)
        self.W = self.W - self.lr * dW

    def lossfunction(self):
        self.forward()
        self.loss = 0.5 * np.sum((self.y_hat - self.y)**2) / self.X.shape[0]

    def predict(self, X_test):
        y_hat_test = np.dot(X_test, self.W)
        return y_hat_test

def RMSE(ypred, yexact):
    return np.sqrt(np.sum((ypred - yexact)**2)/ ypred.shape[0])

myLinearR = LinearR(X_train_norm, y_train, lr = 0.01)
epoch_num = 2000
for i in range(epoch_num):
    myLinearR.forward()
    myLinearR.gradientDescent()
    myLinearR.lossfunction()
    if ((i+1)%20 == 0):
        print('epoch = %d, current loss = %.5f' %(i+1, myLinearR.loss))

y_pred = myLinearR.predict(X_test_norm)
print('RMSE of our model', RMSE(y_pred, y_test))

toc = time.clock()

print('Totol time:' + str((toc-tic))+ 's')

print('===============================Finish===================================')
