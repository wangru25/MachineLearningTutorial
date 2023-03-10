import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
# from nn.Decision_tree import decisiontrees
from nn.Decision_tree_random_features import decisiontrees

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

def one_hot_encoder(y_train, y_test):
    ''' convert label to a vector under one-hot-code fashion '''
    from sklearn import preprocessing
    lb = preprocessing.LabelBinarizer()
    lb.fit(y_train)
    y_train_ohe = lb.transform(y_train)
    y_test_ohe = lb.transform(y_test)
    return y_train_ohe, y_test_ohe

class GBDT():
    def __init__(self, max_depth=5, eta=0.05, n_estimators=10, mode = 'Classification'):
        ''' initalize the decision trees parameters '''
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.eta = eta   # shrinkage, predefined parameters in (0,1]
        self.mode = mode

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.trees = []
        self.alpha = np.arange(-0.01, 0.01, 0.001)
        self.N = self.X.shape[0]
        # y_hat = np.zeros((self.N, self.n_estimators))  # y_hat is the prediction in each tree
        y_pred = np.zeros(self.N).reshape(-1,1)
        for i in range(self.n_estimators):
            if i == 0:
                y_temp = self.y
            else:
                y_temp = residues
            '''
            Note: DO NOT ASSIGN mydt in __init__ PART !!!!!!!!!!!!!!!!!!!!!!!!!!
            '''
            if self.mode == 'Classification':
                mydt = decisiontrees(max_depth=self.max_depth, max_features=8, mode = 'Classification')
            if self.mode == 'Regression':
                mydt = decisiontrees(max_depth=self.max_depth, max_features=3, mode = 'Regression')
            mydt.fit(self.X, y_temp)
            self.trees.append(mydt)
            # y_hat[:,i] = mydt.predict(self.X).astype('float64')
            # y_pred += y_hat[:,i].reshape(-1,1)
            y_pred += mydt.predict(self.X).astype('float64').reshape(-1,1)
            # print(y_pred.ravel())
            residues = (self.y - y_pred).astype('float64')
            #  =========Loss=============
            # loss = np.zeros(len(self.alpha)).reshape(-1,1)
            # for i in range(len(self.alpha)):
            #     loss[i,:] = absolute_loss(self.y, self.alpha[i] * y_pred)
            # min_idx = np.argmin(loss, axis=0)
            # print('min_idx', min_idx)
            # self.lr = self.alpha[min_idx]
            # residues = (self.y - self.eta * self.lr * y_pred).astype('float64')

    def predict(self, X_test):
        num_test_sample = X_test.shape[0]
        y_pred = np.zeros(num_test_sample).reshape(-1,1)
        for tree in self.trees:
            y_pred += tree.predict(X_test).astype('float64').reshape(-1,1)
        if self.mode == 'Classification':
            # y_test_hat = np.rint(y_pred)
            y_test_hat = softmax(y_pred)
        if self.mode == 'Regression':
            y_test_hat = y_pred
        return y_test_hat

def softmax(z):
    exp_value = np.exp(z-np.amax(z, axis=1, keepdims=True)) # for stablility
    # keepdims = True means that the output's dimension is the same as of z
    softmax_scores = exp_value / np.sum(exp_value, axis=1, keepdims=True)
    return softmax_scores

def absolute_loss(y_pred, y):
    return np.sum(np.absolute(y - y_pred),axis=0)

def accuracy(ypred, yexact):
    p = np.array(ypred == yexact, dtype = int)
    return np.sum(p)/float(len(yexact))

def RMSE(ypred, yexact):
    return np.sqrt(np.sum((ypred - yexact)**2)/ ypred.shape[0])

def PCC(y_pred, y_test):
    from scipy import stats
    a = y_test
    b = y_pred
    pcc = stats.pearsonr(a, b)
    return pcc


#==================================Read Dataset=================================
#===================================Classification==============================
# X_train, y_train = read_dataset('MNIST_X_train.csv', 'MNIST_y_train.csv')
# X_test, y_test = read_dataset('MNIST_X_test.csv', 'MNIST_y_test.csv')
X_train, y_train = read_dataset('Digits_X_train.csv', 'Digits_y_train.csv')
X_test, y_test = read_dataset('Digits_X_test.csv', 'Digits_y_test.csv')
y_train_ohe, y_test_ohe = one_hot_encoder(y_train, y_test)


# mygbdt = GBDT(max_depth = 11, eta = 0.5, lr=0.01, n_estimators=10, mode = 'Classification')
# mygbdt.fit(X_train, y_train)
# y_pred = mygbdt.predict(X_test)
# print(y_pred.ravel())
# print(y_test.ravel())

mygbdt = GBDT(max_depth = 11, eta = 0.5, n_estimators=50, mode = 'Classification')
y_pred = np.zeros((y_test_ohe.shape[0], y_test_ohe.shape[1]))
for i in range(10):
    mygbdt.fit(X_train, y_train_ohe[:,i].reshape(-1,1))
    y_pred[:,i] = mygbdt.predict(X_test).ravel()
print(y_pred)

labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
ypred = np.zeros(X_test.shape[0], dtype = int)
for i in range(X_test.shape[0]):
    ypred[i] = labels[np.argmax(y_pred[i,:])]

# print(ypred)
# print(y_test)
print('Accuracy of our model ', accuracy(ypred, y_test.ravel()))

#===================================Regression==================================
# X_train, y_train = read_dataset('airfoil_self_noise_X_train.csv', 'airfoil_self_noise_y_train.csv')
# X_test, y_test = read_dataset('airfoil_self_noise_X_test.csv', 'airfoil_self_noise_y_test.csv')
# # X_train_norm, X_test_norm = normalize_features(X_train, X_test)
#
# mygbdt = GBDT(max_depth = 5, eta = 5, n_estimators=20, mode = 'Regression')
# mygbdt.fit(X_train, y_train)
# y_pred = mygbdt.predict(X_test)
# print(y_pred.ravel())
# print(y_test.ravel())
# print('PCC of our model ', PCC(y_pred.ravel(), y_test.ravel()))
# print('RMSE of our model ', RMSE(y_pred, y_test))

toc = time.clock()

print('Totol time:' + str((toc - tic)) + 's')
print('===============================Finish===================================')
