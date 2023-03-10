import numpy as np
import pandas as pd

class decisiontrees():
    def __init__(self, max_depth=10, current_depth=1):
        ''' initalize the decision trees parameters '''
        self.max_depth = max_depth
        self.current_depth = current_depth
        self.left_tree = None
        self.right_tree = None

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.n_features = X.shape[1]
        self.n_samples = X.shape[0]
        if self.current_depth <= self.max_depth:
            #             print('Current depth = %d' % self.current_depth)
            self.GINI = self.GINI_calculation(self.y)
            self.best_feature_id, self.best_gain, self.best_split_value = self.find_best_split()
            if self.best_gain > 0:
                self.split_trees()

    def predict(self, X_test):
        n_test = X_test.shape[0]
        ypred = np.zeros(n_test, dtype=int)
        for i in range(n_test):
            ypred[i] = self.tree_propogation(X_test[i])
        return ypred

    def tree_propogation(self, feature):
        if self.is_leaf_node():
            #             print(self.y)
            #             print('self.predic',self.predict_label())
            return self.predict_label()
#         print(feature)
        if feature[self.best_feature_id] < self.best_split_value:
            child_tree = self.left_tree
        else:
            child_tree = self.right_tree
        return child_tree.tree_propogation(feature)

    def predict_label(self):
        unique, counts = np.unique(self.y, return_counts=True)
#         print(unique)
#         print(counts)
        label = None
        max_count = 0
        for i in range(unique.size):
            if counts[i] > max_count:
                #                 print(counts[i])
                max_count = counts[i]
                label = unique[i]
#         print('label',label)
        return label

    def is_leaf_node(self):
        return self.left_tree is None

    def split_trees(self):
        # create a left tree
        self.left_tree = decisiontrees(max_depth=self.max_depth,
                                       current_depth=self.current_depth + 1)
        # create a right tree
        self.right_tree = decisiontrees(max_depth=self.max_depth,
                                        current_depth=self.current_depth + 1)
        best_feature_values = self.X[:, self.best_feature_id]
        left_indices = np.where(best_feature_values < self.best_split_value)
        right_indices = np.where(best_feature_values >= self.best_split_value)
        left_tree_X = self.X[left_indices]
        left_tree_y = self.y[left_indices]
        right_tree_X = self.X[right_indices]
        right_tree_y = self.y[right_indices]

        # fit left and right tree
        self.left_tree.fit(left_tree_X, left_tree_y)
        self.right_tree.fit(right_tree_X, right_tree_y)

    def find_best_split(self):
        best_feature_id = None
        best_gain = 0
        best_split_value = None
        for feature_id in range(self.n_features):
            #         for feature_id in range(1, 2):
            current_gain, current_split_value = self.find_best_split_one_feature(
                feature_id)
            if current_gain is None:
                continue
            if best_gain < current_gain:
                best_feature_id = feature_id
                best_gain = current_gain
                best_split_value = current_split_value
        return best_feature_id, best_gain, best_split_value

    def find_best_split_one_feature(self, feature_id):
        '''
            Return information_gain, split_value
        '''
        feature_values = self.X[:, feature_id]
        unique_feature_values = np.unique(feature_values)
        best_gain = 0.0
        best_split_value = None
        if len(unique_feature_values) == 1:
            return best_gain, best_split_value
        for fea_val in unique_feature_values:
            left_indices = np.where(feature_values < fea_val)
            right_indices = np.where(feature_values >= fea_val)
            left_tree_X = self.X[left_indices]
            left_tree_y = self.y[left_indices]

            right_tree_X = self.X[right_indices]
            right_tree_y = self.y[right_indices]
            left_GINI = self.GINI_calculation(left_tree_y)
            right_GINI = self.GINI_calculation(right_tree_y)
#             print(left_GINI)
            # calculate gain
            left_n_samples = left_tree_X.shape[0]
            right_n_samples = right_tree_X.shape[0]
            current_gain = self.GINI - (left_n_samples / self.n_samples * left_GINI +
                                        right_n_samples / self.n_samples * right_GINI)
#             print(self.GINI)
#             print(self.GINI)
            if best_gain < current_gain:
                best_gain = current_gain
                best_split_value = fea_val
        return best_gain, best_split_value

    def GINI_calculation(self, y):
        if y.size == 0 or y is None:
            return 0.0
        unique, counts = np.unique(y, return_counts=True)
        prob = counts / y.size
        return 1.0 - np.sum(prob * prob)
