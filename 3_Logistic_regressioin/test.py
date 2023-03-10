import pandas as pd
import numpy as np
def read_dataset(feature_file,label_file):
    df_x = pd.read_csv(feature_file)
    df_y = pd.read_csv(label_file)
    x = df_x.values
    y = df_y.values
    return x,y
def normalize_features(x_train, x_test):
    from sklearn.preprocessing import StandardScaler
    scalar = StandardScaler()
    scalar.fit(x_train)
    x_train_norm = scalar.transform(x_train)
    x_test_norm = scalar.transform(x_test)
    return x_train_norm, x_test_norm
def accuracy(y_pred, y_exact):
 #   for i in range(y_exact):
  #      if y_exact[i] >= .5:
   #         y_pred = 1
   #     else:
   #         y_pred = 0
   # return y_pred
    p = np.array(y_pred == y_exact, dtype = int)
    return np.sum(p)/(float(len(y_exact)))
x_train, y_train = read_dataset('airfoil_self_noise_X_train.csv', 'airfoil_self_noise_y_train.csv')
x_test, y_test = read_dataset('airfoil_self_noise_X_test.csv', 'airfoil_self_noise_y_test.csv')
x_train_norm, x_test_norm = normalize_features(x_train, x_test)

print(x_train_norm.shape)
print(x_test_norm.shape)
print(y_train.shape)
print(y_test.shape)
class Logistic_Regression:
    def __init__(self, x, y, lr = .01):
            self.x = x
            self.y = y
            self.lr = lr
            self.w = np.zeros((self.x.shape[1], 1))
            self.b = 0.0
            self.m = self.x.shape[0]
    def forward(self):
        self.y_hat = 1 / (1 + np.exp(- np.dot(self.x, self.w) + self.b))
    def gradientDescent(self):
        d = (self.y_hat - self.y) / self.x.shape[0]
        dw = np.dot(self.x.T, d)
        db = np.sum(d, axis = 0, keepdims = True)
        self.w = self.w - self.lr * dw
        self.b = self.b - self.lr *db
    def lossfunction(self):
        self.forward()
        self.loss = 1/self.m * np.sum(-self.y * np.log(self.y_hat))
    def predict(self, x_test):
        y_hat_test = 1 / (1 + np.exp(- np.dot(x_test, self.w) + self.b))
        y_pred = np.zeros((y_hat_test.shape[0],1))
#         print(y_hat_test.shape)
        for i in range(y_hat_test.shape[0]):
            if y_hat_test[i] >= 0.5:
                y_pred[i] = 1
            else:
                y_pred[i] = 0
        return y_pred
myLR = Logistic_Regression(x_train_norm, y_train, 0.01)


epoch_num = 2000
for i in range(epoch_num):
    myLR.forward()
    myLR.gradientDescent()
    myLR.lossfunction()
    if ((i+1)%100 == 0):
        print('epoch = %d, current loss = %.5f'%(i+1, myLR.loss))

y_pred = myLR.predict(X_test_norm)
print('Accuracy of our model', Accuracy(y_pred, y_test.ravel()))
