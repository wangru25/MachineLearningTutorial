import numpy as np
import pandas as pd
import scipy as sp
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

X_train, y_train = read_dataset('MNIST_X_train.csv', 'MNIST_y_train.csv')
X_test, y_test = read_dataset('MNIST_X_test.csv', 'MNIST_y_test.csv')
print(X_train.shape)
X_train_norm, X_test_norm = normalize_features(X_train, X_test)

def one_hot(y_train, y_test):
    from sklearn import preprocessing
    lb = preprocessing.LabelBinarizer()
    lb.fit(y_train)
    y_train_ohe1 = lb.transform(y_train)
    y_test_ohe1 = lb.transform(y_test)
    y_train_ohe = np.where(y_train_ohe1 > 0, y_train_ohe1, -1)
    y_test_ohe = np.where(y_test_ohe1 > 0, y_test_ohe1, -1)
    return y_train_ohe, y_test_ohe

y_train_ohe, y_test_ohe = one_hot(y_train, y_test)
print(y_train_ohe.shape)

def k(Xl, Xi): #Xl of size l x n, Xi of size i x n, where n is number of features
    return np.dot(Xi, Xl.T) #size i x l

class SVM:
    def __init__(self, X, y, lam, alpha):
        self.X = X #size l x n
        self.y = y #size l x 10
        self.lam = lam
        self.alpha = alpha
        self.l = X.shape[0]
        self.W = np.random.rand(self.l,10) #l x 10
        self.b = 0.0
        self.kx = k(self.X, self.X)
    def forward(self):
        self.yhat = np.dot(self.kx.T, self.W) #i x 10, i = l
    def subGradDescent(self):
        dW = (1/(np.linalg.norm(self.W))) * self.W #l x 10
        A =  - np.dot(self.kx.T, self.y) #i x 10, i = l
        dH = self.lam * np.where(A > 0, A, 0)
        self.W = self.W - self.alpha * (dW + dH)
        self.b = self.b - self.alpha * dH
    def predict(self, X_test):
        yhattest = np.dot(k(self.X, X_test), self.W) #i x 10, i = X_test.shape[0]
        for i in range(yhattest.shape[0]):
            result = np.amax(yhattest[i]) #max in each row
            for j in range(yhattest[i].shape[0]):
                if yhattest[i][j]==result:
                    yhattest[i][j] = 1
                else:
                    yhattest[i][j] = -1
        return yhattest
def accuracy(y_pred, y_test):
    p = np.array(y_pred == y_test, dtype = int)
    accuracy = np.sum(p)/float((y_test.size))
    return accuracy

mySVM = SVM(X_train_norm, y_train_ohe, 0.01, 0.1)
epoch_num = 2000
for i in range(epoch_num):
    mySVM.forward()
    mySVM.subGradDescent()
    if(i%20 == 0):
        print(i)
y_pred = mySVM.predict(X_test_norm)
print('accuracy is:', accuracy(y_pred, y_test_ohe))
