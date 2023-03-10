# #--------------------------Imports---------------------------
#
import numpy as np
import pandas as pd
from scipy import stats
from sklearn import datasets



#--------------------------Read Data-------------------------

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

def absolute_loss(y_pred, y):
    return np.sum(np.absolute(y - y_pred), axis=0)



alpha = np.arange(0.01, 1.01, 0.01)
loss = np.zeros(len(alpha)).reshape(-1,1)

y_pred = np.random.randn(y_test.shape[0], y_test.shape[1])

a = absolute_loss(y_pred, y_test)
b = absolute_loss(y_pred, alpha[0]*y_test)
c = y_pred/y_test
print(c)

for i in range(len(alpha)):
    loss[i,:]= absolute_loss(y_test, alpha[i] * y_pred)
# print(type(loss))
# loss = loss.tolist()
print(loss)
min_idx = np.argmin(loss, axis=0)
print(min_idx)
lr = alpha[min_idx]
residues = (y_test - lr * y_pred).astype('float64')

#
# def convert_data(y_train, y_test):
#     ''' convert labels into a list of integers '''
#     labels = np.unique(y_train) # get all the labels of y_train into a list
#     converted_labels = range(len(labels)) # the converted labels will be a list of numbers
#     y_train_conv = np.zeros((len(y_train), 1)) #
#     y_test_conv = np.zeros((len(y_test), 1))
#     for i in range(len(y_train)):
#         labels_index = np.argwhere(labels==y_train[i, 0])[0, 0]
#         y_train_conv[i, 0] = converted_labels[labels_index]
#     for i in range(len(y_test)):
#         labels_index = np.argwhere(labels==y_test[i, 0])[0, 0]
#         y_test_conv[i, 0] = converted_labels[labels_index]
#     return y_train_conv, y_test_conv
#
# #--------------------------Classes----------------------------
#
# class GradientBoostingTree:
#     def __init__(self, max_depth, min_samples_split, n_trees, mode):
#         self.max_depth = max_depth
#         self.min_samples_split = min_samples_split
#         self.n_trees = n_trees
#         self.mode = mode
#         self.trees_list = []
#
#     def fit(self, X, y, lr):
#         self.X = X
#         self.num_samples = self.X.shape[0]
#         self.y = y
#         self.lr = lr
#         pred_temp = 0
#         for tree_num in range(self.n_trees):
#             if tree_num == 0:
#                 y_temp = self.y
#             else:
#                 y_temp = residue
#             tree = DecisionTree.DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split, mode=self.mode)
#             tree.fit(self.X, y_temp)
#             self.trees_list.append(tree)
#             print("loss: ", self.loss())
#             pred_temp += tree.predict(self.X).reshape(-1, 1)
#             residue = self.y - pred_temp
#
#     def predict(self, X_test):
#         num_test_samples = X_test.shape[0]
#         tree_predictions = np.zeros(num_test_samples) # initialize predictions for each tree
#         for tree in self.trees_list:
#             tree_predictions += tree.predict(X_test)
#         return tree_predictions
#
#     def loss(self):
#         return MSE(self.predict(self.X), self.y.ravel())
#
# #--------------------------Functions--------------------------
#
# def accuracy(ypred, yexact):
#     p = np.array(ypred == yexact, dtype = int)
#     return np.sum(p)/float(len(yexact))
#
# def MSE(y_hat, y):
#     return np.mean((y_hat-y)**2)
#
# def MAE(y_hat, y):
#     return np.mean(np.abs(y_hat-y))
#
# #--------------------------Training---------------------------
#
# # data set for regression
# X_train, y_train = read_dataset('~/Downloads/airfoil_dataset/airfoil_self_noise_X_train.csv', '~/Downloads/airfoil_dataset/airfoil_self_noise_y_train.csv')
# X_test, y_test = read_dataset('~/Downloads/airfoil_dataset/airfoil_self_noise_X_test.csv', '~/Downloads/airfoil_dataset/airfoil_self_noise_y_test.csv')
# X_train_norm, X_test_norm = normalize_features(X_train, X_test)
#
# myBooster = GradientBoostingTree(max_depth=5, min_samples_split=1, n_trees=20, mode='regressor')
# myBooster.fit(X_train_norm, y_train, lr=1)
# ypred = myBooster.predict(X_test_norm)
# print("y prediction: ", ypred[:30])
# print("y test: ", y_test.ravel()[:30])
# print("Pearson Correlation: ", stats.pearsonr(ypred, y_test.ravel()))
# print("MSE: ", MSE(ypred, y_test.ravel()))
#
# # data set for regression
# X_train, y_train = read_dataset('airfoil_self_noise_X_train.csv', 'airfoil_self_noise_y_train.csv')
# X_test, y_test = read_dataset('airfoil_self_noise_X_test.csv', 'airfoil_self_noise_y_test.csv')
# X_train_norm, X_test_norm = normalize_features(X_train, X_test)
#
# myBooster = GradientBoostingTree(max_depth=3, min_samples_split=1, n_trees=20, n_features=3, sample_size=50, loss_function='MSE', mode='regressor')
# myBooster.fit(X_train_norm, y_train, lr=0.5)
# ypred = myBooster.predict(X_test_norm)
# print("y prediction: ", ypred[:30])
# print("y test: ", y_test.ravel()[:30])
# print("Pearson Correlation: ", stats.pearsonr(ypred, y_test.ravel()))
#
#
# # # data set for classification
# # iris = datasets.load_iris()
# # index = np.arange(150)
# # np.random.shuffle(index)
# # X = iris.data
# # X = X[index]
# # X_train = X[:120, :]
# # X_test = X[120:, :]
# # y = iris.target
# # y = y[index]
# # y = y.reshape(-1, 1)
# # y_train = y[:120, :]
# # y_test = y[120:, :]
# # X_train_norm, X_test_norm = normalize_features(X_train, X_test)
# #
# # myForest = RandomForest(max_depth=3, min_samples_split=1, n_trees=1000, n_features=2, sample_size=30, mode='classifier')
# # myForest.fit(X_train_norm, y_train)
# # ypred = myForest.predict(X_test_norm)
# # print("y prediction: ", ypred)
# # print("accuracy: ", accuracy(ypred, y_test.ravel()))
