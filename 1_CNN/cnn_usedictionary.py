import numpy as np
import pandas as pd
import time
# import matplotlib as mpl
# mpl.use('TkAgg')
# import matplotlib.pyplot as plt
tic = time.clock()
from nn.convlayers import conv_forward, conv_backward
from nn.poolinglayers import max_pooling_forward, max_pooling_backward
from nn.flattenlayers import flatten_forward, flatten_backward
from nn.fclayers import fc_forward, fc_backward
from nn.activations import relu_forward, relu_backward
from nn.losses import cross_entropy_loss

'''
In this file, we will use dictionary to combine all the seperate layers together
'''


'''
: Structure of this CNN layer: conv1--->pooling--->flatten--->fullyconnect2--->fullyconnect
                                    |                                      |
                                   relu                                   relu
: z : input, with shape (N, C, H, W), in this project, (N,1,28,28)
      N: #of Sampels
      C: input channels, actually the number of colors, usually C = 0,or 1, or 2
      H: Height of input figure
      W: Width of input figure
: K : filter, with shape (C, D, k1, k2), in this project, (1,filters, 3,3)
      C: #of samples
      D: output channels, actually #of filters
      k1: height of filter
      k2: width of filter
: b : bias, with shape (D,)
      D: output channels, actually #of filters
'''

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
    X_train_norm1 = scaler.transform(X_train) # apply normalization on X_train
    X_test_norm1 = scaler.transform(X_test) # we use the same normalization on X_test
    X_train_norm = np.reshape(X_train_norm1,(-1,1,28,28)) # reshape X
    X_test_norm = np.reshape(X_test_norm1,(-1,1,28,28))
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
print(y_train_ohe.shape)
print(y_test_ohe.shape)

weights = {}
filters = 16
weights_scale = 1e-2
fc_nuerons = 500
weights['K1'] = weights_scale * np.random.randn(1, filters, 3, 3).astype(np.float64)
weights['b1']= np.zeros(filters).astype(np.float64)
weights['W2'] = weights_scale * np.random.randn(filters*13*13, fc_nuerons).astype(np.float64)
weights['b2']= np.zeros(fc_nuerons).astype(np.float64)
weights['W3'] = weights_scale * np.random.randn(fc_nuerons, 10).astype(np.float64)
weights['b3'] = np.zeros(10).astype(np.float64)
# initial nuerons and gradients
nuerons={}
gradients={}

def forward(X):
    nuerons['conv1'] = conv_forward(X.astype(np.float64), weights['K1'], weights['b1'])
    nuerons['conv1_relu'] = relu_forward(nuerons['conv1'])
    nuerons['pool1'] = max_pooling_forward(nuerons['conv1_relu'].astype(np.float64), pooling = (2,2))
    nuerons['flatten'] = flatten_forward(nuerons['pool1'])
    nuerons['fc2'] = fc_forward(nuerons['flatten'], weights['W2'], weights['b2'])
    nuerons['fc2_relu'] = relu_forward(nuerons['fc2'])
    nuerons['fc'] = fc_forward(nuerons['fc2_relu'], weights['W3'], weights['b3'])
    return nuerons['fc']

def backward(X, y):
    loss, dy = cross_entropy_loss(nuerons['fc'], y)
    gradients['W3'], gradients['b3'], gradients['fc2_relu'] = fc_backward(dy, weights['W3'], nuerons['fc2_relu'])
    gradients['fc2'] = relu_backward(gradients['fc2_relu'], nuerons['fc2'])
    gradients['W2'], gradients['b2'], gradients['flatten'] = fc_backward(gradients['fc2'], weights['W2'], nuerons['flatten'])
    gradients['pool1'] = flatten_backward(gradients['flatten'], nuerons['pool1'])
    gradients['conv1_relu'] = max_pooling_backward(gradients['pool1'].astype(np.float64), nuerons['conv1_relu'].astype(np.float64),pooling=(2,2))
    gradients['conv1'] = relu_backward(gradients['conv1_relu'], nuerons['conv1'])
    gradients['K1'], gradients['b1'], _ = conv_backward(gradients['conv1'], weights['K1'], X)
    return loss

def accuracy(X, y_exact):
    y_pred = forward(X)
    return np.mean(np.equal(np.argmax(y_pred,axis=-1), np.argmax(y_exact,axis=-1)))

train_num = X_train_norm.shape[0]
def next_batch(batch_size):
    idx = np.random.choice(train_num, batch_size)
    return X_train_norm[idx], y_train_ohe[idx]

from nn.optimizers import SGD
sgd = SGD(weights,lr=0.01, momentum=0.9, decay=5e-6)

batch_size = 10
epoch_num = 2000
for i in range(epoch_num):
    X, y = next_batch(batch_size)
    forward(X)
    backward(X,y)
    sgd.iterate(weights,gradients)
    # if ((i+1)%20 == 0):
    #     print('epoch = %d, current loss = %.5f' % (i+1, myCNN.loss))
toc = time.clock()
print('Accuracy of our model ', accuracy(X_test_norm, y_test_ohe))
print('Totol time:' + str((toc-tic))+ 's')
print('===============================Finish===================================')
