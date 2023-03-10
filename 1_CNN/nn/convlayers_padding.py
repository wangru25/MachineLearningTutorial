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

def conv_forward(z, K, b, padding=(0,0), strides=(1,1)):
    '''
    : Convolutional Forward process
    : padding with 0, only one color.
    : In this project, I will set strides to be 1, if dataset is large, then we can change it be to 2, 3 or more, then some modification needed.
    '''
    padding_z = np.lib.pad(z, ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])), 'constant', constant_values=0)
    N, _, H, W = padding_z.shape
    C, D, k1, k2 = K.shape
    conv_z = np.zeros((N, D, 1 + (H - k1) // strides[0], 1 + (W - k2) // strides[1]))
    for n in np.arange(N):
        for d in np.arange(D):
            for h in np.arange(H - k1 + 1)[::1][::strides[0]]:
                for w in np.arange(W - k2 + 1)[::1][::strides[1]]:
                    conv_z[n, d, h // strides[0] , w // strides[1]] = np.sum(padding_z[n,:, h: h + k1, w: w + k2] * K[:, d, :,:] + b[d])
    return conv_z


def _insert_zeros(dz, strides):
    _, _, H, W = dz.shape
    pz = dz
    if strides[0] > 1:
        for h in np.arange(H - 1, 0, -1):
            for o in np.arange(strides[0] - 1):
                pz = np.insert(pz, h, 0, axis=2)
    if strides[1] > 1:
        for w in np.arange(W - 1, 0, -1):
            for o in np.arange(strides[1] - 1):
                pz = np.insert(pz, w, 0, axis=3)
    return pz

def _remove_padding(z, padding):
    if padding[0] > 0 and padding[1] > 0:
        return z[:, :, padding[0]:-padding[0], padding[1]:-padding[1]]
    elif padding[0] > 0:
        return z[:, :, padding[0]:-padding[0], :]
    elif padding[1] > 0:
        return z[:, :, :, padding[1]:-padding[1]]
    else:
        return z

def conv_backward(next_dz, K, z, strides = (1,1)):
    '''
    : Convolutional Backpropogation process
    : padding with 0, only one color.
    : In this project, I will set strides to be 1, if dataset is large, then we can change it be to 2, 3 or more, then some modification needed.
    '''
    N, C, H, W = z.shape
    C, D, k1, k2 = K.shape
    padding_next_dz = _insert_zeros(next_dz, strides)

    flip_K = np.flip(K, (2, 3))   # rotation 180 degree for k1 * k2 matrix

    T_flip_K = np.transpose(flip_K,(1,0,2,3)) # similar as np.swapaxes(flip_K, 0,1) (Change C, D)
    ppadding_next_dz = np.lib.pad(padding_next_dz, ((0, 0), (0, 0), (k1 - 1, k1 - 1), (k2 - 1, k2 - 1)), 'constant', constant_values=0)

    dz = conv_forward(ppadding_next_dz.astype(np.float32), T_flip_K.astype(np.float32), np.zeros((C,), dtype=np.float32))

    T_z = np.transpose(z, (1,0,2,3))

    dK = conv_forward(T_z.astype(np.float32), padding_next_dz.astype(np.float32), np.zeros((D,), dtype=np.float32))
    db = np.sum(np.sum(np.sum(next_dz, axis=-1), axis=-1), axis=0)

    dz = _remove_padding(dz, padding)


    return dK / N, db / N, dz
