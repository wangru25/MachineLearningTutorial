# import time
import numpy as np
import pandas as pd

# tic = time.clock()


# def read_dataset(feature_file, label_file):
#     ''' Read data set in *.csv to data frame in Pandas'''
#     df_X = pd.read_csv(feature_file)
#     df_y = pd.read_csv(label_file)
#     X = df_X.values  # convert values in dataframe to numpy array (features)
#     y = df_y.values  # convert values in dataframe to numpy array (label)
#     return X, y
#
#
# def normalize_features(X_train, X_test):
#     from sklearn.preprocessing import StandardScaler  # import libaray
#     scaler = StandardScaler()  # call an object function
#     scaler.fit(X_train)  # calculate mean, std in X_train
#     X_train_norm = scaler.transform(X_train)  # apply normalization on X_train
#     X_test_norm = scaler.transform(X_test)
#     return X_train_norm, X_test_norm

class decisiontrees():
    def __init__(self, max_depth=5, current_depth=1, max_features=None, mode = 'Classification'):
        self.max_depth = max_depth
        self.current_depth = current_depth
        self.max_features = max_features   # number of features we choice randomly
        self.mode = mode
        self.left_tree = None
        self.right_tree = None

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
                self.best_feature_id, self.best_score, self.best_split_value = self.find_best_split()
                if self.GINI > 0:
                    self.split_trees()
            if self.mode == 'Regression':
                self.best_feature_id, self.best_score, self.best_split_value = self.find_best_split()
                self.split_trees()

    def predict(self, X_test):
        n_test = X_test.shape[0]
        ypred = np.zeros(n_test, dtype=int)
        for i in range(n_test):
            ypred[i] = self.tree_propogation(X_test[i])
        return ypred

    def tree_propogation(self, feature):
        if self.is_leaf_node():
            return self.predict_label()
        if feature[self.best_feature_id] < self.best_split_value:
            child_tree = self.left_tree
        else:
            child_tree = self.right_tree
        return child_tree.tree_propogation(feature)

    def predict_label(self):
        if self.mode == 'Classification':
            unique, counts = np.unique(self.y, return_counts=True)
            label = None
            max_count = 0
            for i in range(unique.size):
                if counts[i] > max_count:
                    max_count = counts[i]
                    label = unique[i]
            return label
        if self.mode == 'Regression':
            return np.mean(self.y)

    def is_leaf_node(self):
        return self.left_tree is None

    def split_trees(self):
        # create a left tree
        self.left_tree = decisiontrees(max_depth=self.max_depth, current_depth=self.current_depth + 1, mode=self.mode)
        # create a right tree
        self.right_tree = decisiontrees(max_depth=self.max_depth, current_depth=self.current_depth + 1, mode=self.mode)
        best_feature_values = self.X[:, self.best_feature_id]
        left_indices = np.where(best_feature_values < self.best_split_value)[0]
        right_indices = np.where(best_feature_values >= self.best_split_value)[0]
        left_tree_X = self.X[left_indices]
        left_tree_y = self.y[left_indices]
        right_tree_X = self.X[right_indices]
        right_tree_y = self.y[right_indices]
        # fit left and right tree
        self.left_tree.fit(left_tree_X, left_tree_y)
        self.right_tree.fit(right_tree_X, right_tree_y)

    def find_best_split(self):
        best_feature_id = None
        best_score = float('inf')
        best_split_value = None
        idx = np.random.choice(self.n_features, self.max_features, replace=False)
        for feature_id in idx:
            current_score, current_split_value = self.find_best_split_one_feature(feature_id)
            if best_score > current_score:
                best_feature_id = feature_id
                best_score = current_score
                best_split_value = current_split_value
        return best_feature_id, best_score, best_split_value

    def find_best_split_one_feature(self, feature_id):
        '''
            Return information_gain, split_value
        '''
        feature_values = self.X[:, feature_id]
        unique_feature_values = np.unique(feature_values)
        best_score = float('inf')  # For Classification, it's GINI; For Regression, it's Entropy
        best_split_value = None
        if len(unique_feature_values) == 1:
            return best_score, best_split_value
        for fea_val in unique_feature_values:
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
                left_score = self.Entropy_calculation(left_tree_y)
                right_score = self.Entropy_calculation(right_tree_y)
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
        return 1.0 - np.sum(prob ** 2)

    def Entropy_calculation(self, y):
        total_mean = np.mean(y)
        return np.sum((y-total_mean)**2)


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


# #===================================Classification==============================
# # X_train, y_train = read_dataset('MNIST_X_train.csv', 'MNIST_y_train.csv')
# # X_test, y_test = read_dataset('MNIST_X_test.csv', 'MNIST_y_test.csv')
# X_train, y_train = read_dataset('Digits_X_train.csv', 'Digits_y_train.csv')
# X_test, y_test = read_dataset('Digits_X_test.csv', 'Digits_y_test.csv')
#
# max_features = int(np.sqrt(X_train.shape[1])) # This means choose the random feature in each node.
# mydt = decisiontrees(max_depth=11, max_features=max_features, mode = 'Classification')
# mydt.fit(X_train, y_train)
# y_pred = mydt.predict(X_test)
# print('Accuracy of our model ', accuracy(y_pred, y_test.ravel()))
#
# #===================================Regression==================================
# # X_train, y_train = read_dataset('airfoil_self_noise_X_train.csv', 'airfoil_self_noise_y_train.csv')
# # X_test, y_test = read_dataset('airfoil_self_noise_X_test.csv', 'airfoil_self_noise_y_test.csv')
# # X_train_norm, X_test_norm = normalize_features(X_train, X_test)
# #
# # max_features = int(np.sqrt(X_train.shape[1])) # This means choose the random feature in each node.
# # mydt = decisiontrees(max_depth=5, max_features = max_features, current_depth =1, mode = 'Regression')
# # mydt.fit(X_train_norm, y_train)
# # y_pred = mydt.predict(X_test_norm)
# # print('Accuracy of our model ', PCC(y_pred, y_test.ravel()))
# # print('RMSE of our model ', RMSE(y_pred, y_test.ravel()))

# toc = time.clock()
#
# print('Totol time:' + str((toc - tic)) + 's')
#
# print('===============================Finish===================================')
