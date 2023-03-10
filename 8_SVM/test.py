import numpy as np
import pandas as pd

a = np.array([[1,2,3],[5,6,0]])
b = a.argmax(axis=1)
print(a)
print(b)

def read_dataset(feature_file, label_file):
    ''' Read data set in *.csv to data frame in Pandas'''
    df_X = pd.read_csv(feature_file)
    df_y = pd.read_csv(label_file)
    X = df_X.values # convert values in dataframe to numpy array (features)
    y = df_y.values # convert values in dataframe to numpy array (label)
    return X, y


X_train, y_train = read_dataset('MNIST_X_train.csv', 'MNIST_y_train.csv')
X_test, y_test = read_dataset('MNIST_X_test.csv', 'MNIST_y_test.csv')

# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)

def normalize_features(X_train, X_test):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_norm = scaler.transform(X_train)
    X_test_norm = scaler.transform(X_test)
    return X_train_norm, X_test_norm

X_train_norm, X_test_norm = normalize_features(X_train, X_test)


def one_hot_encoder(y_train, y_test):
    ''' convert label to a vector under one-hot-code fashion '''
    from sklearn import preprocessing
    lb = preprocessing.LabelBinarizer()
    lb.fit(y_train)
    y_train_ohe = lb.transform(y_train)
    y_test_ohe = lb.transform(y_test)
    return y_train_ohe, y_test_ohe
# label is 0 -> [1 0 0 0 0 0 0 0 0 0 0 0 ]
# label is 3 -> [0 0 0 1 0 0 0 0 0 0 0 0 ]
y_train_ohe, y_test_ohe = one_hot_encoder(y_train, y_test)
# print(y_test_ohe)



# import numexpr as ne
#
# norm_X1 = np.sum(X_test_norm ** 2, axis = -1)
# norm_X2 = np.sum(X_train_norm ** 2, axis = -1)
# K = ne.evaluate('v * exp(-g * (A + B - 2 * C))', {
#         'A' : norm_X1[:,None],
#         'B' : norm_X2[None,:],
#         'C' : np.dot(X_test_norm, X_train_norm.T),
#         'g' : 10,
#         'v' : 2
# })
# print(K.shape)






# W = np.random.randn(784, 10)
# p = np.dot(X_test_norm, W)
# cond = 1 - y_test_ohe * p
# a = np.argwhere(cond == 1)
# print(cond.shape)
# print(a)

# hinge_loss = np.array([[1,2,1],[3,1,1],[4,1,3],[4,5,2],[5,6,1]])
# y = np.array([[1,4,1],[4,5,0]])
# a = np.where(hinge_loss>3, y, 0)
# print(a)

# b = 1 - y_test_ohe * y_test_ohe
# c = a+b
# print(b)
# print(c)
# print(c.shape)
# def predictor(X, c):
#     return X.dot(c)
#
# def hinge_loss(y_train, y_pred, c, Lambda):
#     # print(y_pred.shape)git
#     hinge_loss_sum = 0.0
#
#     for i in range(y_pred.shape[0]):
#
#         if 1 - y_train[i].dot(y_pred[i]) > 0:
#             hinge_loss = Lambda * (1 - y_train[i].dot(y_pred[i]))
#         else:
#             hinge_loss = 0.0
#         hinge_loss_sum += hinge_loss
#
#     return hinge_loss_sum
#
#
# def loss(y_train, y_pred, c, Lambda):
#
#     L = np.linalg.norm(c) + hinge_loss(y_train, y_pred, c, Lambda)
#
#     return L
#
#
# def sub_gradient_descent(X, y, c, Lambda=0.1, epochs=10, learning_rate=0.001):
#     y = y.reshape(-1, 1) # convert y to a matrix nx1
#     loss_history = [0]*epochs
#
#     for epoch in range(epochs):
#         yhat = predictor(X, c)
#         # print(yhat.shape)
#         loss_history[epoch] = loss(y, yhat, c, Lambda).ravel()
#
#         gradient_sum = np.zeros(c.shape[0])
#
#         for i in range(y.shape[0]):
#             if 1.0 - y[i].dot(yhat[i]) > 0:
#                 g = np.multiply(y[i],X[i])
#                 gradient_sum += g
#
#         c = c - learning_rate * (c * 1.0/np.linalg.norm(c) - Lambda * gradient_sum.reshape(-1,1))
#
#     return c, loss_history
#
# def SVM_binary_train(X_train, y_train):
#     ''' Training our model based on the training data
#         Input: X_train: input features
#                y_train: binary labels
#         Return: coeffs of the logistic model
#     '''
#     coeffs_0 = np.zeros((X_train_norm.shape[1], 1))
#     coeffs_0[0] = 1.0
#     coeffs_grad, history_loss = sub_gradient_descent(X_train, y_train, coeffs_0, Lambda=0.1, epochs=10, learning_rate=0.001)
#     return coeffs_grad
#
#
# def SVM_OVR_train(X_train, y_train):# y_train: one_hot_encoder labels
#     # y_train will have 10 columns
#     weights = []
#     for i in range(y_train.shape[1]): # 10 columns
#         y_train_one_column = y_train[:,i] # pick ith columns
#         weights_one_column = SVM_binary_train(X_train, y_train_one_column)
#         weights.append(weights_one_column)
#     return weights
#
#
# def prediction(weights_list, X_test):
#     i = 0
#     for weights in weights_list:
#         decision_one_column = predictor(X_test, weights)
#         # probabily of one column
#         if i == 0:
#             decision_matrix = decision_one_column
#         else:
#             # combine all decision columns to form a matrix
#             decision_matrix = np.concatenate((decision_matrix, decision_one_column),axis=1)
#         i += 1
#     labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
#     num_test_samples = X_test.shape[0]
#     # find which index gives us the highest probability
#     ypred = np.zeros(num_test_samples, dtype= float)
#     for i in range(num_test_samples):
#         ypred[i] = labels[np.argmax(decision_matrix[i,:])]
#     return ypred
#
#
# weights_list = SVM_OVR_train(X_train_norm, y_train_ohe)
# index = 20
# ypred = prediction(weights_list, X_test_norm)
# # print(ypred)
#
# def accuracy(ypred, yexact):
#     p = np.array(ypred == yexact, dtype = float)
#     return np.sum(p)/float(len(yexact))
#
# ypred = prediction(weights_list, X_test_norm)
# print('Prediction is:\n', ypred)
# print('Accuracy of our model ', accuracy(ypred, y_test.ravel()))
# print(weights_list)
# print(len(weights_list))
