import math
import numpy as np
import pandas as pd
import time

tic = time.perf_counter

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
X_train_norm, X_test_norm = normalize_features(X_train, X_test)

print(X_train_norm.shape)
print(X_test_norm.shape)
print(y_train.shape)
print(y_test.shape)

class LogisticR:
    def __init__(self, X, y, lr = 0.01):
        self.X = X
        self.y = y
        self.lr = lr
        self.m = X.shape[1]
        self.W = np.zeros((self.m,1))
        self.b = 0.0

    def forward(self):
        self.z = np.dot(self.X, self.W) + self.b
        self.y_hat = sigmoid(self.z)
        print(self.y_hat.shape)
        print(self.y.shape)

    def gradientDescent(self):
        d = (self.y_hat - self.y) / self.X.shape[0]
        dW = np.dot(self.X.T, d)
#        dW = np.dot(self.X.T, d) + 0.005*self.W
        db = np.sum(d)
        self.W = self.W - self.lr * dW
        self.b = self.b - self.lr * db

    def lossfunction(self):
        self.forward()
        self.loss = - np.sum(self.y*np.log(self.y_hat)) / self.X.shape[0]
#        self.loss = - np.sum(self.y*np.log(self.y_hat)) / self.X.shape[0] + (0.5*0.005)*(np.sum(self.W**2))

    def predict(self, X_test):
        z = np.dot(X_test, self.W) + self.b
        y_hat_test = sigmoid(z)
        labels = [0,1]
        num_test_samples = X_test.shape[0]
        ypred = np.zeros(num_test_samples,  dtype=int)
        for i in range(num_test_samples):
            if y_hat_test[i] >= 0.5:
                ypred[i] = 1
            else:
                ypred[i] = 0
        return ypred


def sigmoid(z):
    '''This is the finction for sigmoid'''
    return 1/(1+np.exp(-z))

def Accuracy(ypred, yexact):
    p = np.array(ypred == yexact, dtype = int)
    return np.sum(p)/float(len(yexact)) 

myLogisticR = LogisticR(X_train_norm, y_train, lr = 0.1)
epoch_num = 2000
for i in range(epoch_num):
    myLogisticR.forward()
    myLogisticR.gradientDescent()
    myLogisticR.lossfunction()
    if ((i+1)%20 == 0):
        print('epoch = %d, current loss = %.5f' %(i+1, myLogisticR.loss))

y_pred = myLogisticR.predict(X_test_norm)
print('Accuracy of our model', Accuracy(y_pred, y_test.ravel()))

print(y_pred.shape)
print(y_test.ravel().shape)
A = np.array(y_pred == y_test.ravel(), dtype = int)
print(A)

toc = time.process_time()

# print('Totol time:' + str((toc-tic))+ 's')

print('===============================Finish===================================')
