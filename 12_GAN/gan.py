import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
import numpy.ma as ma
import pandas as pd
import math
import time
from nn.activations import *
import seaborn as sns
from PIL import Image, ImageOps
tic = time.perf_counter()



# =================================Data Preprocessing============================
def normalize_features(X_train, X_test):
    from sklearn.preprocessing import StandardScaler  # import libaray
    scaler = StandardScaler()  # call an object function
    scaler.fit(X_train)  # calculate mean, std in X_train
    X_train_norm = scaler.transform(X_train)  # apply normalization on X_train
    # we use the same normalization on X_test
    X_test_norm = scaler.transform(X_test)
    return X_train_norm, X_test_norm

f1 = open('/Data/train-images-idx3-ubyte')
f2 = open('/Data/train-labels-idx1-ubyte')
f3 = open('/Data/t10k-images-idx3-ubyte')
f4 = open('/Data/t10k-labels-idx1-ubyte')

X_train_loaded = np.fromfile(file=f1, dtype=np.uint8)
y_train_loaded = np.fromfile(file=f2, dtype=np.uint8)
X_train = (X_train_loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float32) /  127.5 - 1).reshape(60000, 784)
y_train = y_train_loaded[8:].reshape((60000,1)).astype(np.int32)
X_test_loaded = np.fromfile(file=f3, dtype=np.uint8)
y_test_loaded = np.fromfile(file=f4, dtype=np.uint8)
X_test = (X_test_loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float32) /  127.5 - 1).reshape(10000, 784)
y_test = y_test_loaded[8:].reshape((10000,1)).astype(np.int32)

X_train_norm, X_test_norm = normalize_features(X_train, X_test)

print(X_train_norm.shape)
print(X_test_norm.shape)
print(y_train.shape)
print(y_test.shape)


def DataReader(X, y, numbers):
    newXtrain = []
    newytrain = []
    for idx in range(X.shape[0]):
        if y[idx] == numbers:
            newXtrain.append(X[idx])
            newytrain.append(y[idx])
    return np.array(newXtrain), np.array(newytrain), X.shape[0]

def plot(fake_img):
    sns.set_style("dark")
    N = fake_img.shape[0]
    H = int(N / 5)
    fig, axes = plt.subplots(H, 5, figsize=(5, H))
    for i in range(H):
        for j in range(5):
            axes[i][j].imshow(fake_img[5*i+j].reshape(28, 28), cmap='gray_r', interpolation='nearest')
            axes[i][j].set_xticks([])
            axes[i][j].set_yticks([])
    plt.savefig('fake1.png')

class GAN():
    def __init__(self, X, numbers, epochs, lr, decay, batch_size, H1, H2):
        '''
        Structure of this GAN:
            z      -->    h0   -->  fake out   <--> image    -->  h0    -->  out
            (10,100)   (100,128)    (128,784)    (128,784)    (100,128)   (100,1)
        '''
        self.X = X
        self.numbers = numbers
        self.epochs = epochs
        self.lr = lr
        self.decay = decay
        self.batch_size = batch_size   # 10
        self.H1 = H1        # 100
        self.H2 = H2        # 128
        self.D_in = X.shape[1]   # 784, 28*28

        self.G_W0 = np.random.randn(self.H1, self.H2) / np.sqrt(self.H1)
        self.G_b0 = np.zeros((1, self.H2))
        self.G_W1 = np.random.randn(self.H2, self.D_in) / np.sqrt(self.H2)
        self.G_b1 = np.zeros((1, self.D_in))

        self.D_W0 = np.random.randn(self.D_in, self.H2) / np.sqrt(self.D_in)
        self.D_b0 = np.zeros((1, self.H2))
        self.D_W1 = np.random.randn(self.H2, 1) / np.sqrt(self.H2)
        self.D_b1 = 0.0

    def Generator(self, z):
        self.z = z.reshape(self.batch_size, -1)
        self.G_z1 = np.dot(self.z, self.G_W0) + self.G_b0
        self.G_f1 = tanh(self.G_z1)
        self.G_z2 = np.dot(self.G_f1, self.G_W1) + self.G_b1
        self.G_f2 = sigmoid(self.G_z2)
        self.G_hat = self.G_f2.reshape(self.batch_size, 28, 28)
        return self.G_z2, self.G_hat

    def Discriminator(self, img):
        self.img = img.reshape(self.batch_size, -1)
        self.D_z1 = np.dot(self.img, self.D_W0) + self.D_b0
        self.D_f1 = tanh(self.D_z1)
        self.D_z2 = np.dot(self.D_f1, self.D_W1) + self.D_b1   # logit output
        self.D_f2 = sigmoid(self.D_z2)
        self.D_hat = self.D_f2   # sigmoid output
        return self.D_z2, self.D_hat

    def Gen_Backprop(self, fake_logit, fake_output, fake_input):
        '''
        Input:
		    fake_logit : (10, 784) Fake logit value before sigmoid activation function (generated input)
		    fake_output : (10, 1)  Discriminator output in range 0~1 (generated input)
		    fake_input : (10, 784) Fake input image fed into the discriminator
        '''
        fake_input = fake_input.reshape(self.batch_size, -1)  # 10*784
        G_fake_loss = -1.0 / (fake_output + 1e-8)  # (10,1)

        # calculate the gradients from the end of the discriminator
        d1_fake_D = G_fake_loss * sigmoid(fake_logit, derivative=True)  # 10*1
        d0_fake_D = np.dot(d1_fake_D, self.D_W1.T) * tanh(self.D_z1, derivative=True) # 10*128
        d_fake = np.dot(d0_fake_D, self.D_W0.T)  # 10*784
        d1_fake_G = np.dot(d0_fake_D, self.D_W0.T) * sigmoid(self.G_z2, derivative=True) # 10*784
        dW1_fake = np.dot(self.G_f1.T, d1_fake_G)
        db1_fake = np.sum(d1_fake_G, axis = 0, keepdims=True)

        d0_fake_G = np.dot(d1_fake_G, self.G_W1.T) * tanh(self.G_z1, derivative=True) # 10*128
        dW0_fake = np.dot(self.z.T, d0_fake_G)
        db0_fake = np.sum(d0_fake_G, axis = 0, keepdims=True)

        # Calculate generator gradients, loss = -log(D(G(z)))
        self.G_W1 = self.G_W1 - self.lr * dW1_fake
        self.G_b1 = self.G_b1 - self.lr * db1_fake
        self.G_W0 = self.G_W0 - self.lr * dW0_fake
        self.G_b0 = self.G_b0 - self.lr * db0_fake


    def Dis_Backprop(self, real_logit, real_output, real_input, fake_logit, fake_output, fake_input):
        '''
        Input:
            real_logit : (10, 128) Real logit value before sigmoid activation function (real input)
		    real_output : (10, 1)  Discriminator output in range 0~1 (real input)
		    real_input : (10, 784) Real input image fed into the discriminator
		    fake_logit : (10, 784) Fake logit value before sigmoid activation function (generated input)
		    fake_output : (10, 1)  Discriminator output in range 0~1 (generated input)
		    fake_input : (10, 784) Fake input image fed into the discriminator
        '''
        real_input = real_input.reshape(self.batch_size, -1)  # 10*784
        fake_input = fake_input.reshape(self.batch_size, -1)  # 10*784

        # Real image backpropgation, loss = -log(D(x))
        D_real_loss = -1.0 / (real_output + 1e-8)        # (10, 1)
        D_fake_loss = 1.0 / (1.0 - fake_output + 1e-8)  # (10, 1)

        d1_real = D_real_loss * sigmoid(real_logit, derivative=True)  # (10, 1)
        dW1_real =  np.dot(self.D_f1.T, d1_real)
        db1_real = np.sum(d1_real, axis = 0, keepdims=True)

        d0_real = np.dot(d1_real, self.D_W1.T) * tanh(self.D_z1, derivative=True)
        dW0_real = np.dot(real_input.T, d0_real)
        db0_real = np.sum(d0_real, axis = 0, keepdims=True)

        # Fake image backpropgation, loss = -log(1 - D(G(z)))
        d1_fake = D_fake_loss * sigmoid(fake_logit, derivative=True)  # (10, 1)
        dW1_fake =  np.dot(self.D_f1.T, d1_fake)
        db1_fake = np.sum(d1_fake, axis = 0, keepdims=True)

        d0_fake = np.dot(d1_fake, self.D_W1.T) * tanh(self.D_z1, derivative=True)
        dW0_fake = np.dot(fake_input.T, d0_fake)
        db0_fake = np.sum(d0_fake, axis = 0, keepdims=True)

        # The total gradient for real and fake images
        dW1 = dW1_real + dW1_fake
        db1 = db1_real + db1_fake
        dW0 = dW0_real + dW0_fake
        db0 = db0_real + db0_fake

        # Upadate gradient for different batch_size
        self.D_W1 = self.D_W1 - self.lr * dW1
        self.D_b1 = self.D_b1 - self.lr * db1
        self.D_W0 = self.D_W0 - self.lr * dW0
        self.D_b0 = self.D_b0 - self.lr * db0


    def train(self):
        Xtrain, ytrain, train_size = DataReader(X_train_norm, y_train, self.numbers)
        np.random.shuffle(Xtrain)
        batch_idx = train_size // self.batch_size
        Gloss = []
        Dloss = []
        Real_Ave = []
        Fake_Ave = []
        for epoch in range(self.epochs):
            for idx in range(batch_idx):
                train_batch = Xtrain[idx * self.batch_size : (idx + 1) * self.batch_size]
                if train_batch.shape[0] != self.batch_size:
                    break
                # z = np.random.uniform(-1,1,[self.batch_size, self.H1])
                z = np.random.randn(self.batch_size, self.H1)


                # Forward
                G_logits, fake_img = self.Generator(z)
                D_real_logits, D_real_output = self.Discriminator(train_batch)
                D_fake_logits, D_fake_output = self.Discriminator(fake_img)

                D_loss = - 0.5 * np.log(D_real_output + 1e-8) - 0.5 * np.log(1 - D_fake_output + 1e-8)
                G_loss = - np.log(D_fake_output + 1e-8)

			    # Backward
                self.Dis_Backprop(D_real_logits, D_real_output, train_batch, D_fake_logits, D_fake_output, fake_img)
                self.Gen_Backprop(D_fake_logits, D_fake_output, fake_img)
				# # train generator twice?
				# G_logits, fake_img = self.Generator(z)
				# D_fake_logits, D_fake_output = self.Discriminator(fake_img)
				# self.Gen_Backprop(D_fake_logits, D_fake_output, fake_img)

                # print("Epoch [%d] Step [%d] G Loss:%.4f D Loss:%.4f Real Ave.: %.4f Fake Ave.: %.4f"%(epoch, idx, np.mean(G_loss), np.mean(D_loss), np.mean(D_real_output), np.mean(D_fake_output)))
            Gloss.append(np.mean(G_loss))
            Dloss.append(np.mean(D_loss))
            Real_Ave.append(np.mean(D_real_output))
            Fake_Ave.append(np.mean(D_fake_output))


			#update learning rate every epoch
            self.lr = self.lr * (1.0 / (1.0 + self.decay * epoch))
        plot(fake_img)
        # plt.plot(Dloss,label='D')
        # plt.plot(Gloss,label='G')
        # plt.title('Minmax loss of the discriminator and the generator',fontweight='bold')
        # plt.grid()
        # plt.legend()
        # plt.xlabel('Iteration')
        # plt.ylabel('AVE')
        # plt.show()





# numbers = [7]
gan = GAN(X_train, numbers=[0], epochs=2000, lr=1e-2, decay=1e-2, batch_size=10, H1=128, H2=512)
gan.train()
toc = time.perf_counter()
print(("Elapsed time: %.1f [min]" % ((toc-tic)/60)))
