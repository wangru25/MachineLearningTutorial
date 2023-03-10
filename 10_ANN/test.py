import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def readcsv(feature, label):  #read csv
    x = pd.read_csv(feature)
    y = pd.read_csv(label)
    x_values = x.values
    y_values = y.values
    return x_values, y_values

def norm(x_train, x_test):  # norm values
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train_norm = scaler.transform(x_train)
    x_test_norm = scaler.transform(x_test)
    return x_train_norm, x_test_norm

def RELU(x):
    return x * (x>0)


def RELU_derivate(x):
    return 1. * (x>0)


class ANN():
    def __init__(self, x, y, h=0.05, alpha=0.01, H1=500, H2=100):
        self.x = x
        self.y = y
        self.h = h
        self.m1, self.n1 = x.shape
        self.m2, self.n2 = y.shape
        self.loss = 0.0
        self.alpha = alpha
        self.weight1 = np.random.randn(self.n1, H1)
        self.weight2 = np.random.randn(H1, H2)
        self.weight3 = np.random.randn(H2, self.n2)
        self.b1 = np.zeros((1, H1))
        self.b2 = np.zeros((1, H2))
        self.b3 = np.zeros((1, self.n2))


    def forward(self):
        self.z1 = np.dot(self.x, self.weight1) + self.b1
        self.layer1 = RELU(self.z1)
        self.z2 = np.dot(self.layer1, self.weight2) + self.b2
        self.layer2 = RELU(self.z2)
        self.z3 = np.dot(self.layer2, self.weight3) + self.b3
        self.output = RELU(self.z3)

    def backward(self):
        delta_weight3_dot = (self.output - self.y) * RELU_derivate(self.z3)
        delta_weight3 = np.dot(self.layer2.T, delta_weight3_dot)
        delta_b3 = np.dot(np.ones((1, self.m1)), delta_weight3_dot)

        delta_weight2_dot = (np.dot(delta_weight3_dot,
                                    self.weight3.T)) * RELU_derivate(self.z2)
        delta_weight2 = np.dot(self.layer1.T, delta_weight2_dot)
        delta_b2 = np.dot(np.ones((1, self.m1)), delta_weight2_dot)

        delta_weight1_dot = np.dot(delta_weight2_dot,
                                   self.weight2.T) * RELU_derivate(self.z1)
        delta_weight1 = np.dot(self.x.T, delta_weight1_dot)
        delta_b1 = np.dot(np.ones((1, self.m1)), delta_weight1_dot)

        self.weight3 = self.weight3 - self.h * (delta_weight3 + self.alpha * self.weight3)
        self.b3 = self.b3 - self.h * delta_b3
        self.weight2 = self.weight2 - self.h * (delta_weight2 + self.alpha * self.weight2)
        self.b2 = self.b2 - self.h * delta_b2
        self.weight1 = self.weight1 - self.h *( delta_weight1 + self.alpha * self.weight1)
        self.b1 = self.b1 - self.h * delta_b1


    def predictor(self, x_test):
        self.layer1_pre = np.dot(x_test, self.weight1) + self.b1
        self.layer2_pre = np.dot(self.layer1_pre, self.weight2) + self.b2
        self.y_pre = np.dot(self.layer2_pre, self.weight3) + self.b3
        print(self.y_pre.shape)
        return self.y_pre

    # def loss_function(self):
    #     self.forward()
    #     self.loss = np.sum((self.output-self.y)*(self.output-self.y))
    #     return self.loss


def one_hot_encoder(y_train, y_test):
    ''' convert label to a vector under one-hot-code fashion '''
    from sklearn import preprocessing
    lb = preprocessing.LabelBinarizer()
    lb.fit(y_train)
    y_train_ohe = lb.transform(y_train)
    y_test_ohe = lb.transform(y_test)
    return y_train_ohe, y_test_ohe

def one_hot_encoder(y_train, y_test):
    ''' convert label to a vector under one-hot-code fashion '''
    from sklearn import preprocessing
    lb = preprocessing.LabelBinarizer()
    lb.fit(y_train)
    y_train_ohe = lb.transform(y_train)
    y_test_ohe = lb.transform(y_test)
    return y_train_ohe, y_test_ohe


def one_hot_decoder(y_test):
    y_test_real = y_test.argmax(axis=1)
    return y_test_real.reshape(-1, 1)


def accuracy(y_real, y_pre):
    z = y_pre - y_real
    error = np.sum(z == 0)
    m = y_pre.shape[0]
    return error / float(m)


x_train, y_train = readcsv('MNIST_X_train.csv','MNIST_y_train.csv')
x_test, y_test = readcsv('MNIST_X_test.csv','MNIST_y_test.csv')
x_train_norm, x_test_norm = norm(x_train, x_test)
y_train_ohe, y_test_ohe = one_hot_encoder(y_train, y_test)

h = 0.0001
alpha = 0.01

H1 = 500
H2 = 100

ANN_class = ANN(x_train_norm, y_train_ohe, h, alpha, H1, H2)

epoch = 200
for i in range(epoch):
    ANN_class.forward()
    ANN_class.backward()
    # ANN_class.loss_function()
    # print(loss)

y_test_ohe_hat = ANN_class.predictor(x_test_norm)
y_hat = one_hot_decoder(y_test_ohe_hat)
print(y_hat.shape)
print(y_test.shape)
acc = accuracy(y_hat, y_test)
print(acc)

# x = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
# y=np.array([[0,1,1,0]]).T
# ann1=ANN(x,y,0.1,0.1,3,2)
# for i in range(200):
#     ann1.forward()
#     ann1.backward()
#     loss=ann1.loss_function()
#     print(loss)




# print('Accuracy of our model ', accuracy(y_pred, y_test.ravel()))
