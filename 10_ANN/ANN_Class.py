import numpy as np
import pandas as pd
import math

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


X_train, y_train = read_dataset('MNIST_X_train.csv', 'MNIST_y_train.csv')
X_test, y_test = read_dataset('MNIST_X_test.csv', 'MNIST_y_test.csv')
X_train_norm, X_test_norm = normalize_features(X_train, X_test)
y_train_ohe, y_test_ohe = one_hot_encoder(y_train, y_test)

print(X_train_norm.shape)
print(X_test_norm.shape)
print(y_train.shape)
print(y_train_ohe.shape)
print(y_test_ohe.shape)
print(y_train)
print(y_train_ohe)

class ANN:
    def __init__(self, X, y, hidden_layer_nn_1=500, hidden_layer_nn_2=100, lr=0.01, Lambda = 0.1):
        self.X = X # 64 features in first hidden layer
        self.y = y # labels (targets) in one-hot-encoder
        self.hidden_layer_nn_1 = hidden_layer_nn_1 # number of neuron in the 1st hidden layer
        self.hidden_layer_nn_2 = hidden_layer_nn_2 # number of neuron in the 1st hidden layer
        # In this example, we only consider 1 hidden layer
        self.lr = lr # learning rate
        self.Lambda =Lambda
        # Initialize weights
        self.nn = X.shape[1] # number of neurons in the inpute layer
        self.W1 = np.random.randn(self.nn, hidden_layer_nn_1) / np.sqrt(self.nn)
        self.b1 = np.zeros((1, hidden_layer_nn_1)) # double parentheses
        self.W2 = np.random.randn(hidden_layer_nn_1, self.hidden_layer_nn_2) / np.sqrt(hidden_layer_nn_1)
        self.b2 = np.zeros((1, self.hidden_layer_nn_2))
        self.output_layer_nn = y.shape[1]
        self.W3 = np.random.randn(hidden_layer_nn_2, self.output_layer_nn) / np.sqrt(hidden_layer_nn_2)
        self.b3 = np.zeros((1, self.output_layer_nn))

    def feed_forward(self):
        self.z1 = np.dot(self.X, self.W1) + self.b1
        self.f1 = relu_forward(self.z1)
        self.z2 = np.dot(self.f1, self.W2) + self.b2
        self.f2 = relu_forward(self.z2)
        self.z3 = np.dot(self.f2, self.W3) + self.b3
        self.y_hat = softmax(self.z3)

    def back_propagation(self):
        # $d_3 = \hat{y}-y$
        d3 = self.y_hat - self.y
        # dL/dW3 = f_1^T d_2
        dW3 = np.dot(self.f2.T, d3) + self.Lambda * self.W3
        # dL/b_3 = d_3.dot(1)$
        db3 = np.sum(d3, axis=0, keepdims=True) # axis =0 : sum along the vertical axis
        # d_2 = relu_backward(z_2) * d_3 W_3^T
        d2 = relu_backward(self.z2) * (d3.dot((self.W3).T))
        # np.dot(self.f1.T, d2)/self.X.shape[0] + self.Lambda * self.W2
        dW2 = np.dot(self.f1.T, d2) + self.Lambda * self.W2
        # dL/b_3 = d_3.dot(1)$
        db2 = np.sum(d2, axis=0, keepdims=True) # axis =0 : sum along the vertical axis
        # d_1 = relu_backward(z_1)*W_2^T*relu_backward(z_2) * (\hat{y}-y)W_3^T
        # d1 = (self.W2.T).dot(relu_backward((self.z1).T)) * relu_backward(self.z2) * (d3.dot((self.W3).T))
        d1 = relu_backward(self.z1) * (d2.dot((self.W2).T))
        # dL/dW_1} = x^T d_z
        dW1 = np.dot(self.X.T, d1) + self.Lambda * self.W1
        # dL/db_1 = d_1
        db1 = np.sum(d1, axis=0, keepdims=True) # axis =0 : sum along the vertical axis

        # Update the gradident descent
        self.W1 = self.W1 - self.lr * dW1
        self.b1 = self.b1 - self.lr * db1
        self.W2 = self.W2 - self.lr * dW2
        self.b2 = self.b2 - self.lr * db2
        self.W3 = self.W3 - self.lr * dW3
        self.b3 = self.b3 - self.lr * db3

    def cross_entropy_loss(self):

        self.feed_forward()
#         self.loss = 0.5*np.sum((self.y_hat - self.y)**2)/self.y.shape[0] + (0.5*self.Lambda)*(np.sum(self.W1**2)+np.sum(self.W2**2)+np.sum(self.W3**2))
        self.loss = -np.sum(self.y * np.log(self.y_hat + 1e-8))/self.y.shape[0] + (0.5*self.Lambda)*(np.sum(self.W1**2)+np.sum(self.W2**2)+np.sum(self.W3**2))
#         self.loss = -np.sum(self.y * np.log(self.y_hat))/self.y.shape[0] + (0.5*self.Lambda)*(np.sum(self.W1**2)+np.sum(self.W2**2)+np.sum(self.W3**2))
    def predict(self, X_test):

        z1 = np.dot(X_test, self.W1) + self.b1
        f1 = relu_forward(z1)
        z2 = np.dot(f1, self.W2) + self.b2
        f2 = relu_forward(z2)
        z3 = np.dot(f2, self.W3) + self.b3
        y_hat_test = softmax(z3)
        labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        num_test_samples = X_test.shape[0]
        ypred = np.zeros(num_test_samples, dtype=int)
        for i in range(num_test_samples):
            ypred[i] = labels[np.argmax(y_hat_test[i,:])]
        return ypred

# def relu_forward(z):
#     '''This is the finction for ReLU'''
#     z[z < 0] = 0
#     return z
#
# def relu_backward(z):
#     '''This is the derivative of ReLU'''
#     z[z <= 0] = 0
#     z[z > 0] = 1
#     return z
def relu_forward(z):
    '''This is the finction for ReLU'''
    return z * (z > 0)

def relu_backward(z):
    '''This is the derivative of ReLU'''
    return 1. * (z > 0)


def softmax(z):
    exp_value = np.exp(z-np.amax(z, axis=1, keepdims=True)) # for stablility
    # keepdims = True means that the output's dimension is the same as of z
    softmax_scores = exp_value / np.sum(exp_value, axis=1, keepdims=True)
    return softmax_scores

def accuracy(ypred, yexact):
    p = np.array(ypred == yexact, dtype = int)
    return np.sum(p)/float(len(yexact))

import time
tic  = time.time()
myNN = ANN(X_train_norm, y_train_ohe, hidden_layer_nn_1=500, hidden_layer_nn_2=100, lr=0.0001, Lambda=0.01)
epoch_num = 200
for i in range(epoch_num):
    myNN.feed_forward()
    myNN.back_propagation()
    myNN.cross_entropy_loss()
    if ((i+1)%20 == 0):
        print('epoch = %d, current loss = %.5f' % (i+1, myNN.loss))

y_pred = myNN.predict(X_test_norm)
print('Accuracy of our model ', accuracy(y_pred, y_test.ravel()))

toc = time.time()
print('Totol time:' + str((toc-tic))+ 's')
