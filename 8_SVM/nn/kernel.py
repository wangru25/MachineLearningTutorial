import numpy as np
import numpy.linalg as la


def linear_kernel(x, y):
    return np.dot(x, y)

def poly_kernel(x, y):
    alpha = 1
    r = 1
    d = 1
    return (alpha + np.dot(x,y)) ** d

def RBF_kernel(x, y):
    sigma = 2
    mu = 1
    return np.exp(-(la.norm(x - y) / sigma) ** mu)

def tanget_kernel(x, y):
    gamma = 2
    r = 1
    return np.tanh(gamma * np.dot(x, y) + r)

def sigmoid_kernel(x, y):
    gamma = 2
    return 1 / (1 + np.exp(-gamma * np.dot(x,y)))

def gauss_kernel(x, y):
    sigma = 2
    exponent = - la.norm(x-y) ** 2 / (2 * sigma ** 2)
    return np.exp(exponent)

    norms = np.sum(data_matrix**2, axis=-1)
    gram = data_matrix.dot(data_matrix.T)
    K = np.exp((-1/(tau**2))*(-2*gram + norms.reshape((1, -1)) + norms.reshape((-1, 1))))
    return K
