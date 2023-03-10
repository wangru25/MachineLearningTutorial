import numpy as np
import math



def relu_forward(z):
    '''This is the finction for ReLU'''
    return z * (z > 0)

def relu_backward(next_dz, z):
    '''This is the derivative of ReLU'''
    dz = np.where(np.greater(z,0), next_dz, 0)
    return dz

def tanh_forward(z):
    '''This is the finction for tanh'''
    return np.tanh(z)

def tanh_backward(dz):
    '''This is the derivative of tanh'''
    return 1. - np.tanh(dz) * np.tanh(dz)

def sigmoid_forward(z):
    '''This is the finction for sigmoid'''
    return 1/(1+np.exp(-z))

def sigmoid_backward(dz):
    '''This is the derivative of sigmoid'''
    return np.exp(-dz)/ ( (1 + np.exp(-dz)) * (1 + np.exp(-dz)) )
