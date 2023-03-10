import math
import numpy as np
import pandas as pd
'''
: Flatten Layer:
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

def flatten_forward(z):
    """
    flatten N-D array, feedforward
    z: (N,d1,d2,..)
    """
    N = z.shape[0]
    return np.reshape(z, (N, -1))


def flatten_backward(next_dz, z):
    """
    flatten N-D array, backpropagation
    """
    return np.reshape(next_dz, z.shape)
