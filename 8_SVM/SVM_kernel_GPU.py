import math
import time
import random
import torch
import numpy as np
import pandas as pd
import numexpr as ne
# from nn.kernel import linear_kernel, poly_kernel, RBF_kernel, tanget_kernel, sigmoid_kernel, gauss_kernel

dtype = torch.float
device = torch.device('cuda:0')

tic = time.clock()
torch.manual_seed(7)


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


#
# def one_hot_encoder(y_train, y_test):
#     ''' convert label to a vector under one-hot-code fashion '''
#     from sklearn import preprocessing
#     lb = preprocessing.LabelBinarizer()
#     lb.fit(y_train)
#     y_train_ohe = lb.transform(y_train)
#     y_test_ohe = lb.transform(y_test)
#     return y_train_ohe, y_test_ohe
#
def one_hot_encoder(y_train, y_test):
    ''' convert label to a vector under one-hot-code fashion '''
    from sklearn import preprocessing
    lb = preprocessing.LabelBinarizer()
    lb.fit(y_train)
    y_train_ohe1 = lb.transform(y_train)
    y_test_ohe1 = lb.transform(y_test)
    y_train_ohe = np.where(y_train_ohe1 > 0, y_train_ohe1, -1)
    y_test_ohe = np.where(y_test_ohe1 > 0, y_test_ohe1, -1)
    return y_train_ohe, y_test_ohe


X_train, y_train = read_dataset('MNIST_X_train.csv', 'MNIST_y_train.csv')
X_test, y_test = read_dataset('MNIST_X_test.csv', 'MNIST_y_test.csv')
X_train_norm, X_test_norm = normalize_features(X_train, X_test)
y_train_ohe, y_test_ohe = one_hot_encoder(y_train, y_test)

Xtrain = torch.tensor(X_train, device=device, dtype=dtype)
Xtest = torch.tensor(X_test, device=device, dtype=dtype)
ytrain = torch.tensor(y_train_ohe, device=device, dtype=dtype)
ytest = torch.tensor(y_test_ohe, device=device, dtype=dtype)

print(Xtrain.shape)
print(Xtest.shape)
print(ytrain.shape)
print(ytest.shape)


class SVM():
    def __init__(self, X, y, kernel = 'RBF', r=1, polyconst=1, gamma = 10, degree = 2, lr=0.01, Lambda=0.01, batch_size =10):
        '''
        Input:
        - X: shape (N, D)
        - y: shape (N,C) C = 10
        - W shape (D,C)
        '''
        self.kernel = kernel
        self.kf = {'linear': self.linear, 'RBF': self.RBF, 'poly': self.poly}
        self.r = r
        self.polyconst = polyconst
        self.gamma = gamma
        self.degree = degree
        self.X = X
        self.y = y
        self.lr = lr
        self.Lambda = Lambda
        self.batch_size = batch_size
        self.m = X.shape[0]  # numbers of samples
        self.n = X.shape[1]  # numbers of features
        self.p = y.shape[1]
        self.W = torch.randn(self.m, self.p, device=device, dtype=dtype) / self.m  # 784 by 10
        self.b = torch.zeros(1, self.p, device=device, dtype=dtype)   # 1 by 10

    def linear(self, x, y):
        return torch.mm(x, torch.t(y)) * self.polyconst

    def poly(self, x, y):
        return (torch.mm(x, torch.t(y)) * self.polyconst + self.r) ** self.degree

    def RBF(self, x, y):
        norm_x = torch.sum(x ** 2, dim = 1)
        norm_y = torch.sum(y ** 2, dim = 1)
        K = ne.evaluate('exp(-g * (A + B - 2 * C))', {
                'A' : norm_x[:,None],
                'B' : norm_y[None,:],
                'C' : torch.mm(x, torch.t(y)),
                'g' : self.gamma
        })
        return K
        # return np.exp(-1.0 * self.gamma * np.dot(np.subtract(x,y).T,np.subtract(x,y)))
        # return np.exp(-self.gamma * np.sum((x[..., None, :] - y) ** 2, axis=2))
        # return np.exp(-1.0 * self.gamma * np.subtract(x,y) * np.subtract(x,y))


    def forward(self):
        # self.K = np.zeros((self.m, self.m))
        # for i in range(self.m):
        #     for j in range(self.m):
        #         self.K[i,j] = self.kf[self.kernel](self.X[i], self.X[j])
        # print(self.K.shape)
        self.K = self.kf[self.kernel](self.X, self.X)
        # print(self.K)
        self.y_hat = torch.mm(self.K, self.W) + self.b # 2000 by 10
        self.cond = 1 - self.y * self.y_hat

    def loss(self):
        self.forward()
        Q = torch.zeros_like(self.y)
        self.hinge_loss = torch.where(self.cond > 0, self.y, Q)  # 2000 by 10
        self.loss = torch.norm(self.W, dim=0) + self.Lambda * self.hinge_loss

    def sub_gradient_descent(self):
        Q = torch.zeros_like(self.y)
        y = torch.where(self.cond > 0, self.y, Q)
        y_b = torch.where(self.cond > 0,  - self.y, Q)
        dW = (1/torch.norm(self.W, dim=0)) * self.W - \
            self.Lambda * torch.t(torch.mm(torch.t(y), self.K))
        db = torch.sum(self.Lambda * y_b , dim = 0)
        self.W = self.W - self.lr * dW
        self.b = self.b - self.lr * db

    # def mini_batch_gradient(self):
    #     X = self.X
    #     y = self.y
    #     self.K = self.kf[self.kernel](self.X, self.X)
    #     self.K, self.y = mini_batch(self.K, self.y, self.batch_size)
    #     self.forward()
    #     self.sub_gradient_descent()
    #     self.X = X
    #     self.y = y

    def predict(self, X_test):
        K = self.kf[self.kernel](X_test, self.X)
        # print(K.shape)
        y_hat_test = torch.mm(K, self.W) + self.b
        labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        num_test_samples = X_test.shape[0]
        ypred = torch.zeros(num_test_samples, device=device, dtype=dtype)
        for i in range(num_test_samples):
            # ypred[i] = labels[np.argmax(y_hat_test[i, :])]
            ypred[i] = labels[y_hat_test[i,:].max(0)[1]]
        return ypred


# def mini_batch(X,y,batch_size):
#     N = X.shape[0]
#     idx = np.random.choice(N, batch_size)
#     return X[idx], y[idx]


def accuracy(ypred, yexact):
    p = torch.tensor(ypred == yexact, device=device, dtype=dtype)
    return torch.sum(p) / float(len(yexact))


mySVM = SVM(X_train_norm, y_train_ohe, kernel='linear', r=1, polyconst=0.01, gamma = 0.001, degree = 2, lr=0.001, Lambda=0.01, batch_size =10)
epoch_num = 1000
for i in range(epoch_num):
    mySVM.loss()
    mySVM.sub_gradient_descent()
    # mySVM.mini_batch_gradient()
    if ((i+1)% 10 == 0):
        print('epoch = %d, current loss = %.5f' % (i+1, mySVM.loss))

y_pred = mySVM.predict(X_test_norm)
print(y_pred)
print(y_test.ravel())
print('Accuracy of our model ', accuracy(y_pred, y_test.ravel()))

toc = time.clock()

print('Totol time:' + str((toc - tic)) + 's')
print('===============================Finish===================================')
