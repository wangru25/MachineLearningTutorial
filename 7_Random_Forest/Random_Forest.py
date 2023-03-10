import time
import numpy as np
import pandas as pd
from scipy import stats
from Decision_tree_random_features import decisiontrees


tic = time.clock()


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
    X_test_norm = scaler.transform(X_test)
    return X_train_norm, X_test_norm


class RandomForest():
    def __init__(self, n_estimators=10, mode = 'Classification'):
        self.n_estimators = n_estimators  # the number of trees in a forest
        self.mode = mode
        if self.mode == 'Classification':
            self.mydt = decisiontrees(max_depth=10, max_features=8, mode = 'Classification')
        if self.mode == 'Regression':
            self.mydt = decisiontrees(max_depth=5, max_features=3, mode = 'Regression')
        # if max_feature = 'auto':
        #     self.max_feature = self.n_features
        # if max_feature = 'sqrt':
        #     self.max_feature = np.sqrt(self.n_features)
        # if max_feature = 'log2':
        #     self.max_feature = np.log2(self.n_features)

    def bagging(self, X, y):
        self.X = X
        self.y = y
        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]
        self.idx = np.random.randint(0, self.X.shape[0], size=(
            self.n_estimators, self.X.shape[0]))
        self.X_bag = self.X[self.idx]
        self.y_bag = self.y[self.idx]

    def predict_RF(self, X_test):
        n_test = X_test.shape[0]
        y_pred_bag = np.zeros((self.n_estimators, n_test))
        for i in range(self.n_estimators):
            self.mydt.fit(self.X_bag[i], self.y_bag[i].ravel())
            y_pred_bag[i, :] = self.mydt.predict(X_test)
        if self.mode == 'Classification':
            y_pred = stats.mode(y_pred_bag)[0]
        if self.mode == 'Regression':
            y_pred = np.mean(y_pred_bag, axis = 0)
        return y_pred


def accuracy(ypred, yexact):
    p = np.array(ypred == yexact, dtype=int)
    return np.sum(p) / float(len(yexact))

def RMSE(ypred, yexact):
    return np.sqrt(np.sum((ypred - yexact)**2)/ ypred.shape[0])

def PCC(y_pred, y_test):
    from scipy import stats
    a = y_test
    b = y_pred
    accuracy = stats.pearsonr(a, b)
    return accuracy

# ===================================Classification==============================
X_train, y_train = read_dataset('MNIST_X_train.csv', 'MNIST_y_train.csv')
X_test, y_test = read_dataset('MNIST_X_test.csv', 'MNIST_y_test.csv')
# X_train, y_train = read_dataset('Digits_X_train.csv', 'Digits_y_train.csv')
# X_test, y_test = read_dataset('Digits_X_test.csv', 'Digits_y_test.csv')

myRF = RandomForest(n_estimators=10, mode='Classification')
myRF.bagging(X_train, y_train.ravel())
y_pred = myRF.predict_RF(X_test)
print('Accuracy of our model ', accuracy(y_pred, y_test.ravel()))

# # ===================================Regression==================================
# X_train, y_train = read_dataset('airfoil_self_noise_X_train.csv', 'airfoil_self_noise_y_train.csv')
# X_test, y_test = read_dataset('airfoil_self_noise_X_test.csv', 'airfoil_self_noise_y_test.csv')
# X_train_norm, X_test_norm = normalize_features(X_train, X_test)

# myRF = RandomForest(n_estimators=10, mode='Regression')
# myRF.bagging(X_train_norm, y_train.ravel())
# y_pred = myRF.predict_RF(X_test_norm)
# # print(y_pred)
# # print(y_test.ravel())
# print('PCC of our model ', PCC(y_pred, y_test.ravel()))
# print('RMSE of our model ', RMSE(y_pred, y_test.ravel()))

toc = time.clock()
print('Totol time:' + str((toc - tic)) + 's')
print('===============================Finish===================================')
