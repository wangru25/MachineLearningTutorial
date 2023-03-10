import math
import numpy as np
import pandas as pd
'''
: Pooling Layer:
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
: K : filter, with shape (C, D, k1, k2), in this project, (1,filters, 3,3)
      C: #of samples
      D: output channels, actually #of filters
      k1: height of filter
      k2: width of filter
: b : bias, with shape (D,)
      D: output channels, actually #of filters
'''

def max_pooling_forward(z, pooling, strides = (2,2)):
    """
    : Max Pooling Forward process
    : pooling: Usually is the 2 by 2 matrix, just like filter.
    """
    N, C, H, W = z.shape
    s0 = strides[0]
    s1 = strides[1]
    out_h = (H - pooling[0]) // s0 + 1
    out_w = (W - pooling[1]) // s1 + 1
    pool_z = np.zeros((N, C, out_h, out_w))

    for n in np.arange(N):
        for c in np.arange(C):
            for i in np.arange(out_h):
                for j in np.arange(out_w):
                    pool_z[n, c, i, j] = np.max(z[n, c, i * s0 : i*s0 + pooling[0], j*s1 : j*s1 + pooling[1]])
    return pool_z


def max_pooling_backward(next_dz, z, pooling, strides=(2, 2)):
    """
    : Max Pooling Backrward process
    : next_dz: loss of max pooling(about loss funtion)
    : z: (N,C,H,W)
    : pooling: (k1,k2)
    """
    N, C, H, W = z.shape
    _, _, out_h, out_w = next_dz.shape
    pool_dz = np.zeros_like(z)
    s0 = strides[0]
    s1 = strides[1]
    for n in np.arange(N):
        for c in np.arange(C):
            for i in np.arange(out_h):
                for j in np.arange(out_w):
                    flat_idx = np.argmax(z[n, c, i*s0: i*s0 + pooling[0], j*s1 : j*s1 + pooling[1]])
                    h_idx = s0 * i + flat_idx // pooling[1]
                    w_idx = s0 * j + flat_idx % pooling[1]
                    pool_dz[n, c, h_idx, w_idx] += next_dz[n, c, i, j]
    return pool_dz
