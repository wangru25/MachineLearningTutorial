import math
import numpy as np
import pandas as pd
# from nn.flip import flip
'''
: Convolutional Layer:
: z : input, with shape (N, C, H, W), in this project, (N,1,28,28)
      N: #of Sampels
      C: input channels, actually the number of colors, usually C = 0,or 1, or 2
      H: Height of input figure
      W: Width of input figure
: next_dz : The gradient of the output Convolutional layer, with shape (N, D, H', W')
      N: #of Sampels
      C: input channels, actually the number of colors, usually C = 0,or 1, or 2
      H': Height
      W': Width
: K : filter, with shape (C, D, k1, k2), in this project, (1,filters, 5,5)
      C: #of samples
      D: output channels, actually #of filters
      k1: height of filter
      k2: width of filter
: b : bias, with shape (D,)
      D: output channels, actually #of filters
'''

def conv_forward(z, K, b):
    '''
    : Convolutional Forward process
    : padding with 0, only one color.
    : In this project, I will set strides to be 1, if dataset is large, then we can change it be to 2, 3 or more, then some modification needed.
    '''
    N, _, H, W = z.shape
    C, D, k1, k2 = K.shape
    conv_z = np.zeros((N, D, 1 + (H - k1), 1 + (W - k2)))
    for n in np.arange(N):
        for d in np.arange(D):
            for h in np.arange(H - k1 + 1)[::1]:
                for w in np.arange(W - k2 + 1)[::1]:
                    conv_z[n, d, h , w] = np.sum(z[n,:, h: h + k1, w: w + k2] * K[:, d, :,:] + b[d])
    return conv_z


def conv_backward(next_dz, K, z, strides = (1,1)):
    '''
    : Convolutional Backpropogation process
    : padding with 0, only one color.
    : In this project, I will set strides to be 1, if dataset is large, then we can change it be to 2, 3 or more, then some modification needed.
    '''
    N, C, H, W = z.shape
    C, D, k1, k2 = K.shape

    flip_K = np.flip(K, (2, 3))   # rotation 180 degree for k1 * k2 matrix
    # print('flip_K', flip_K.shape)

    T_flip_K = np.transpose(flip_K,(1,0,2,3)) # similar as np.swapaxes(flip_K, 0,1) (Change C, D)
    # print('T_flip_K', T_flip_K.shape)
    dz = conv_forward(next_dz.astype(np.float32), T_flip_K.astype(np.float32), np.zeros((C,), dtype=np.float32))
    # print('dz', dz.shape)

    T_z = np.transpose(z, (1,0,2,3))
    # print('T_z', T_z.shape)
    # print('next_dz',next_dz.shape)

    dK = conv_forward(T_z.astype(np.float32), next_dz.astype(np.float32), np.zeros((D,), dtype=np.float32))
    # print('dK', dK.shape)
    db = np.sum(np.sum(np.sum(next_dz, axis=-1), axis=-1), axis=0)


    return dK / N, db / N, dz
