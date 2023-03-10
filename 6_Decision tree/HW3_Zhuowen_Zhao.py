#!/usr/bin/env python3

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

class decisiontrees():
    def __init__(self, max_depth=5, current_depth=1, max_features=None, mode = 'Classification'):
        self.max_depth = max_depth
        self.current_depth = current_depth
        self.max_features = max_features
        self.mode = mode
        self.left_tree = None
        self.right_tree = None

#---------------------  functions for labeling(prediction/testing)-----------------------
    def predict(self, X_test):
        n_test = X_test.shape[0]
        ypred = np.zeros(n_test, dtype=int)
        for i in range(n_test):
            ypred[i] = self.tree_propogation(X_test[i])
        return ypred

    def tree_propogation(self, feature):                                                  # go down the tree
        if self.left_tree is None:                                                        
            return self.predict_label()                                                   # predict label here
        if feature[self.best_feature_id] < self.best_split_value:                         # which leaf to go based on the established parameters
            child_tree = self.left_tree
        else:
            child_tree = self.right_tree
        return child_tree.tree_propogation(feature)                                       # keep going down

    def predict_label(self):
        if self.mode == 'Classification':
            unique, counts = np.unique(self.y, return_counts=True)                        # self.y is the label after training? where store each category?
            label = None
            max_count = 0
            for i in range(unique.size):
                if counts[i] > max_count:
                    max_count = counts[i]
                    label = unique[i]
            return label                                                                  # label with the most frequent labels in y
        elif self.mode == 'Regression':
            return np.mean(self.y)

#-------------------functions for tree designing and training parameters------------------
    def find_best_split(self):                                                            # decide what features to choose for split and criterion, and the GINI score
        best_feature_id = None
        best_score = float('inf')
        best_split_value = None
        idx = np.random.choice(self.n_features, self.max_features, replace=False)         # find the best feature indices from the random choice of features
        for feature_id in idx:
            current_score, current_split_value = self.find_best_split_one_feature(feature_id)
            if best_score > current_score:
                best_feature_id = feature_id
                best_score = current_score
                best_split_value = current_split_value
        return best_feature_id, best_score, best_split_value

    def find_best_split_one_feature(self, feature_id):                                    # find the best criterion of one feature by score (classification)
        '''
            Return information_gain, split_value
        '''
        feature_values = self.X[:, feature_id]
        unique_feature_values = np.unique(feature_values)
        best_score = float('inf')                                                         # For Classification, it's GINI; For Regression, it's Entropy
        best_split_value = None                                                           # parameter
        if len(unique_feature_values) == 1:                                               # cannot be used the feature to split, therefor return inf score
            return best_score, best_split_value
        for fea_val in unique_feature_values:                                             # find criterion for grouping from existing feature values
            left_indices = np.where(feature_values < fea_val)[0]
            right_indices = np.where(feature_values >= fea_val)[0]

            left_tree_X = self.X[left_indices]
            left_tree_y = self.y[left_indices]

            right_tree_X = self.X[right_indices]
            right_tree_y = self.y[right_indices]

            left_n_samples = left_tree_y.shape[0]
            right_n_samples = right_tree_y.shape[0]

            if left_n_samples == 0 or right_n_samples == 0:
                continue
            if self.mode == 'Classification':
                left_score = self.GINI_calculation(left_tree_y)
                right_score = self.GINI_calculation(right_tree_y)
            if self.mode == 'Regression':
                left_score = self.SD_calculation(left_tree_y)
                right_score = self.SD_calculation(right_tree_y)
            current_score = (left_n_samples * left_score) / self.n_samples  + (right_n_samples * right_score) / self.n_samples 
            if best_score > current_score:
                best_score = current_score
                best_split_value = fea_val
        return best_score, best_split_value

    def GINI_calculation(self, y):
        if y.size == 0 or y is None:
            return 0.0
        unique, counts = np.unique(y, return_counts=True)
        prob = counts / y.size
        return 1.0 - np.sum(prob ** 2)                                                    # the probability of impurity

    def SD_calculation(self, y):
        if y.size == 0 or y is None:
          return 0.0
        return np.std(y)

#--------------------main ( tree designing and training parameters)----------------------
    def fit(self, X, y):
        self.X = X
        self.y = y
        if self.max_features == None:
            self.max_features = X.shape[1]
        self.n_features = X.shape[1]
        self.n_samples = X.shape[0]
        if self.current_depth <= self.max_depth:
            if self.mode == 'Classification':
                # print('Current depth = %d' % self.current_depth)
                self.GINI = self.GINI_calculation(self.y)
                self.best_feature_id, self.best_score, self.best_split_value = self.find_best_split() # determine which feature and what criterion
                if self.GINI > 0:
                    self.split_trees()                                                    # split data/trees based on find_best_split()
            if self.mode == 'Regression':
                self.SD = self.SD_calculation(self.y)
                self.best_feature_id, self.best_score, self.best_split_value = self.find_best_split() # determine which feature and what criterion
                if self.SD > 0:
                    self.split_trees()

    def split_trees(self):
        # create a left tree
        self.left_tree = decisiontrees(max_depth=self.max_depth, current_depth=self.current_depth + 1, mode=self.mode)
        # create a right tree
        self.right_tree = decisiontrees(max_depth=self.max_depth, current_depth=self.current_depth + 1, mode=self.mode)
        best_feature_values = self.X[:, self.best_feature_id]                             # split based on only one feature at a time(node) 
        left_indices = np.where(best_feature_values < self.best_split_value)[0]
        right_indices = np.where(best_feature_values >= self.best_split_value)[0]
        left_tree_X = self.X[left_indices]
        left_tree_y = self.y[left_indices]
        right_tree_X = self.X[right_indices]
        right_tree_y = self.y[right_indices]
        # fit left and right tree
        self.left_tree.fit(left_tree_X, left_tree_y)                                      # keeping going forward along each subbranch: feed left_tree the 'left data',right tree the 'right data' 
        self.right_tree.fit(right_tree_X, right_tree_y)                                   # ??? also updating parameters, how to store


class RandomForest():
    def __init__(self, n_estimators=10, mode = 'Classification'):
        self.n_estimators = n_estimators                                                  # the number of trees in a forest
        self.mode = mode
        if self.mode == 'Classification':
            self.mydt = decisiontrees(max_depth=10, max_features=8, mode = 'Classification') # initialize decision tree
        if self.mode == 'Regression':
            self.mydt = decisiontrees(max_depth=5, max_features=3, mode = 'Regression')

    def bagging(self, X, y):
        self.X = X
        self.y = y
        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]
        self.idx = np.random.randint(0, self.X.shape[0], size=(self.n_estimators, self.X.shape[0])) # bootstrap training data for n number of trees
        self.X_bag = self.X[self.idx]
        self.y_bag = self.y[self.idx]

#------------------------------- training and testing -----------------------------------
    def predict_RF(self, X_test):
        n_test = X_test.shape[0]
        y_pred_bag = np.zeros((self.n_estimators, n_test))
        for i in range(self.n_estimators):
            self.mydt.fit(self.X_bag[i], self.y_bag[i].ravel())                           # call decision tree to train one tree at a time
            y_pred_bag[i, :] = self.mydt.predict(X_test)                                  # prediction of test data after training for each tree at time
        if self.mode == 'Classification':
            y_pred = stats.mode(y_pred_bag)[0]                                            # final prediction as most modal(frequent) vote among all trees (let all trees fully grow), default column wise
        if self.mode == 'Regression':
            y_pred = np.mean(y_pred_bag,axis=0)
        return y_pred

def read_dataset(feature_file, label_file):
    ''' Read data set in *.csv to data frame in Pandas'''
    df_X = pd.read_csv(feature_file)
    df_y = pd.read_csv(label_file)
    X = df_X.values  # convert values in dataframe to numpy array (features)
    y = df_y.values  # convert values in dataframe to numpy array (label)
    return X, y

def accuracy(ypred, yexact):                                                              #only for classification
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
    
#===================================Classification==============================
# X_train, y_train = read_dataset('MNIST_X_train.csv', 'MNIST_y_train.csv')
# X_test, y_test = read_dataset('MNIST_X_test.csv', 'MNIST_y_test.csv')
# X_train, y_train = read_dataset('Digits_X_train.csv', 'Digits_y_train.csv')
# X_test, y_test = read_dataset('Digits_X_test.csv', 'Digits_y_test.csv')
# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)
# 
# myRF = RandomForest(n_estimators=10, mode='Classification')                               # initialize RandomForest class
# myRF.bagging(X_train, y_train.ravel())
# y_pred = myRF.predict_RF(X_test)
# print('Accuracy of our model ', accuracy(y_pred, y_test.ravel()))

#===================================Regression==================================
X_train, y_train = read_dataset('airfoil_self_noise_X_train.csv', 'airfoil_self_noise_y_train.csv')
X_test, y_test = read_dataset('airfoil_self_noise_X_test.csv', 'airfoil_self_noise_y_test.csv')
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

myRF = RandomForest(n_estimators=10, mode='Regression')                                   # initialize RandomForest class
myRF.bagging(X_train, y_train.ravel())
y_pred = myRF.predict_RF(X_test)
print('RMSE of our regression model {}'.format(RMSE(y_pred, y_test.ravel())))
print('Pearson coeffecient of our regression model {}'.format(PCC(y_pred, y_test.ravel())[0]))

