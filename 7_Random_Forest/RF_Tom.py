import numpy as np
import pandas as pd
def read_dataset(feature_file, label_file):
    ''' Read data set in *.csv to data frame in Pandas'''
    df_X = pd.read_csv(feature_file)
    df_y = pd.read_csv(label_file)
    X = df_X.values  # convert values in dataframe to numpy array (features)
    y = df_y.values  # convert values in dataframe to numpy array (label)
    return X, y

def accuracy(ypred, yexact):
    p = np.array(ypred == yexact, dtype=int)
    return np.sum(p) / float(len(yexact))

def pearson(y_pred, y_test):
    from scipy import stats
    a = y_test.ravel()
    b = y_pred.ravel()
    accuracy = stats.pearsonr(a,b)
    return accuracy

def RMSE(ypred, ytest):
    return np.sqrt(np.sum((ytest-ypred)**2) /(ytest.shape[0]))


class decisiontrees():
    def __init__(self, max_depth=5, current_depth=1,mode = 'Classification', rf = False):        
        ''' initalize the decision trees parameters '''
        self.max_depth = max_depth
        self.current_depth = current_depth
        self.left_tree = None
        self.right_tree = None
        self.mode = mode
        self.rf = rf
       
    def fit(self, X, y, feats=None):
        self.X = X
        self.y = y        
        self.n_features = X.shape[1]
        self.n_samples = X.shape[0]
        if self.rf:
            self.featUsed = np.zeros(self.n_features)
            for i in range(feats.shape[0]):
                self.featUsed[feats[i]] = 1
        if self.current_depth <= self.max_depth:
#             print('Current depth = %d' % self.current_depth)
            if self.mode == 'Classification':
                self.GINI = self.GINI_calculation(self.y)
                self.best_feature_id, self.best_gain, self.best_split_value = self.find_best_split()
                if self.best_gain > 0:
                    self.split_trees()
                    self.featUsed[self.best_feature_id] = 0
            if self.mode == 'Regression':
                self.STD = np.std(self.y)
                self.best_feature_id, self.best_gain, self.best_split_value = self.find_best_split()
                if self.best_gain > 0:
                    self.split_trees()
               
       
    def predict(self, X_test): # predict all the samples in the test set by tree propagation for each sample
        n_test = X_test.shape[0]
        ypred = np.zeros(n_test, dtype=int)  
        for i in range(n_test):
            ypred[i] = self.tree_propogation(X_test[i])            
        return ypred
               
    def tree_propogation(self, feature): # goes through the tree to predict one sample
        if self.is_leaf_node(): # if the node is a leaf, we predict the label
            return self.predict_label()
        if feature[self.best_feature_id] < self.best_split_value:  # choose which path to take based on best split value
            child_tree = self.left_tree
        else:
            child_tree = self.right_tree
        return child_tree.tree_propogation(feature) # recursively call the tree propagation
   
    def predict_label(self): # predicts the label in a leaf node
        if self.mode == 'Classification':
            unique, counts = np.unique(self.y, return_counts=True) # find the labels in the leaf node and their counts
            label = None  # initialize label and largest count
            max_count = 0
            for i in range(unique.size):  # go through the counts of each label
                if counts[i] > max_count: # find the label that has the most samples associated with it
                    max_count = counts[i]
                    label = unique[i]
            return label
        if self.mode == 'Regression':
            labelSum = np.sum(self.y)
            return labelSum/self.y.size
   
    def is_leaf_node(self):
        return self.left_tree is None
   
    def split_trees(self):
        # create a left tree
        self.left_tree = decisiontrees(max_depth = self.max_depth, current_depth = self.current_depth + 1, mode = self.mode)
        self.right_tree = decisiontrees(max_depth = self.max_depth, current_depth = self.current_depth + 1, mode = self.mode)
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
            current_gain, current_split_value = self.find_best_split_one_feature(feature_id)
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
        if self.rf and self.featUsed[feature_id]==0:
            return 0.0, None
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
            left_STD = np.std(left_tree_y)
            right_STD = np.std(right_tree_y)

            # calculate gain
            if self.mode == 'Classification':
                left_n_samples = left_tree_X.shape[0]
                right_n_samples = right_tree_X.shape[0]
                current_gain = self.GINI - (left_n_samples/self.n_samples * left_GINI + right_n_samples/self.n_samples * right_GINI)
            if self.mode == 'Regression':
                left_n_samples = left_tree_X.shape[0]
                right_n_samples = right_tree_X.shape[0]
                current_gain = self.STD - (left_n_samples/self.n_samples * left_STD + right_n_samples/self.n_samples * right_STD)
            if best_gain < current_gain:
                best_gain = current_gain
                best_split_value = fea_val
        return best_gain, best_split_value                                    
           
    def GINI_calculation(self, y):
        if y.size == 0 or y is None:
            return 0.0
        unique, counts = np.unique(y, return_counts=True)
        prob = counts/y.size
        return 1.0 - np.sum(prob*prob)
   
class randomforrest():
    def __init__(self, nTree, nFeatures, mode):
        self.nTree = nTree
        self.nFeatures = nFeatures
        self.mode = mode
        self.trees = np.empty(1, dtype=decisiontrees)
        self.trees[0] = decisiontrees(max_depth = self.nFeatures, mode = self.mode, rf = True)
        self.createTrees()
    def createTrees(self):
        if self.trees.shape[0] < self.nTree:
            TreeAdd = np.empty(1, dtype=decisiontrees)
            TreeAdd[0] = decisiontrees(max_depth = self.nFeatures, mode = self.mode, rf = True)
            self.trees = np.append(self.trees, TreeAdd, axis = 0)
            self.createTrees()
    def fitTrees(self, X, y):
        for t in range(self.nTree):
            self.fitOneTree(self.trees[t], X, y)
            print('tree fitted')
    def fitOneTree(self, tree, X, y):
        feats = np.sort(np.random.permutation(X.shape[1])[:self.nFeatures])
        tree.fit(X, y, feats)
    def predict(self, Xtest):
        if self.mode=='Classification':
            pred = np.zeros((self.nTree,Xtest.shape[0]), dtype=int)
            ypred = np.zeros((self.nTree,Xtest.shape[0]), dtype=int)
            for t in range(self.nTree):
                pred[t] = self.trees[t].predict(Xtest)
            for s in range(pred.shape[1]):
                ypred[:, s] = self.classificationA(pred[:,s])
        if self.mode=='Regression':
            pred = self.trees[0].predict(Xtest)
            for t in range(self.nTree-1):
                pred = pred + self.trees[t+1].predict(Xtest)
            print(pred.shape)
            ypred = pred/self.nTree
        return ypred
    def classificationA(self, yonerow):
        unique, counts = np.unique(yonerow, return_counts=True) # find the labels in the leaf node and their counts
        label = None  # initialize label and largest count
        max_count = 0
        for i in range(unique.size):  # go through the counts of each label
            if counts[i] > max_count: # find the label that has the most samples associated with it
                max_count = counts[i]
                label = unique[i]
        return label
   
X_train, y_train = read_dataset('airfoil_self_noise_X_train.csv', 'airfoil_self_noise_y_train.csv')
X_test, y_test = read_dataset('airfoil_self_noise_X_test.csv', 'airfoil_self_noise_y_test.csv')
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

myrf = randomforrest(5, 4, mode='Regression')
myrf.fitTrees(X_train, y_train)
y_pred = myrf.predict(X_test)
ypred = y_pred.reshape(-1, 1)
print(ypred.shape)
print('correlation of our model ', pearson(ypred, y_test))
print('RMSE', RMSE(ypred, y_test))
