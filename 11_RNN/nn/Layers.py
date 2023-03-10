import math
import numpy as np
import pandas as pd
from nn.activations import tanh, softmax, sigmoid

# ===============================RNN_Cell_Forward================================

def rnn_cell_forward(Xt, h_prev, parameters):
    '''
    Reference : Andrew Ng, Deep Learning Homework Week3 
    RNN Cell:
    Input:
        - Xt: Input data at timestep "t", shape: (N, D)
            : N : #of samples.
            : D : #of input examples.
        - h_prev: Hidden state at timestep "t-1", shape: (N, H)
            : N : #of samples.
            : H : #of hidden neurans
        - parameters: a dictionary containing:
            : Wx : Weight matrix multiplying the input, shape (D, H)
            : Wh : Weight matrix multiplying the hidden state, shape (H, H)
            : Wy : Weight matrix relating the hidden-state to the output, shape (H, M), M is 10 in MNIST Problem.
            : bh  : Bias, shape (1, H)
            : by  : Bias, shape (1, M)
    Returns:
        - h_next : next hidden state, shape (N, H)
        - yt_pred: prediction at timestep "t", shape (N, M)
        - cache  : tuple of values needed for the backward pass,
                   contains (h_next, h_prev, Xt, parameters)
    '''
    # Retrieve parameters from "parameters"
    Wx = parameters['Wx']
    Wh = parameters['Wh']
    Wy = parameters['Wy']
    bh = parameters['bh']
    by = parameters['by']

    # compute next activation state using the formula given above
    h_next = tanh(np.dot(Xt, Wx) + np.dot(h_prev, Wh) + bh)
    yt_pred = softmax(np.dot(h_next, Wy) + by)

    # store values we need for Backward propagation in cache
    cache = (h_next, h_prev, Xt, parameters)

    return h_next, yt_pred, cache


def rnn_cell_backward(dh_next, cache):
    '''
    Backward of cell_RNN:
    Input:
        - dh _next : Gradient of loss wrt next hidden state
        - caches   : Output of rnn_cell_forward
    Returns:
        - dX      : Gradient of the input data, shape (N, D)
        - dh_prev : Gradient of previous hidden state, shape (N, H)
        - dWx     : Gradient of the input-to-hidden Weight matrix, shape (D, H)
        - dWh     : Gradient of the hidden-to-hidden Weight matrix, shape (H, H)
        - dbh     : Gradient of the bias, shape (1,H)
    '''

    (h_next, h_prev, Xt, parameters) = cache
    Wx = parameters['Wx']
    Wh = parameters['Wh']
    Wy = parameters['Wy']
    bh = parameters['bh']
    by = parameters['by']

    dtanh = (1 - h_next**2) * dh_next   # (N, H)

    dXt = np.dot(dtanh, Wx.T)   # (N, D)
    dWx = np.dot(Xt.T, dtanh)   # (D, H)

    dh_prev = np.dot(dtanh, Wh.T) # (N, H)
    dWh = np.dot(h_prev.T, dtanh) # (H, H)


    dbh = np.sum(dtanh, axis = 0, keepdims=True) # (1, H)

    gradients = {'dXt': dXt, 'dh_prev': dh_prev,
                 'dWx': dWx, 'dWh': dWh, 'dbh': dbh}

    return gradients

# ===============================RNN_Forward====================================


def rnn_forward(X, h0, parameters):
    '''
    Forward Layers of RNN:
    Input:
        - X : Input data for every time-step, shape: (N, D, T)
            : N : #of samples.
            : D : #of input examples.
            : T : the length of the input sequence
        - h0: Initial hidden state, shape: (N, H)
            : N : #of samples.
            : H : #of hidden neurans.
        - parameters: a dictionary containing:
            : Wx : Weight matrix multiplying the input, shape (D, H)
            : Wh : Weight matrix multiplying the hidden state, shape (H, H)
            : Wy : Weight matrix relating the hidden-state to the output, shape (H, M)
            : bh  : Bias, shape (1, H)
            : by  : Bias, shape (1, M), M = 10
    Returns:
        - h     : Hidden states for every time-step, shape (N, H, T)
        - y_pred: Predictions for every time-step, shape (N, M, T)
        - caches  : tuple of values needed for the backward pass,
                   contains (list of caches, X)
    '''

    # Initial "caches" which will contain the list of all caches
    caches = []

    # Retrieve dimensions from shape of X and Wy
    N, D, T = X.shape
    H, M = parameters['Wy'].shape

    # Initialize 'h' and 'y'
    h = np.zeros((N, H, T))
    y_pred = np.zeros((N, M, T))

    # Initialize h_next
    h_next = h0
    for t in range(T):
        h_next, yt_pred, cache = rnn_cell_forward(X[:, :, t], h_next, parameters)
        h[:, :, t] = h_next
        y_pred[:, :, t] = yt_pred
        caches.append(cache)

    # Store values needed for backward propagation in cache
    caches = (caches, X)

    return h, y_pred, caches


def rnn_backward(dh, caches):
    '''
    Backward Layers of RNN:
    Input:
        - dh    : Upstream gradients of all hidden states, shape (N, H, T)
                : N : #of samples.
                : D : #of input examples.
                : T : the length of the input sequence
        - caches: Tuple containing information from the forward pass (rnn_forward)
    Returns:
        - dh    : Gradient wrt the input data, shape (N, H, T)
        - dh0   : Gradient wrt the initial hidden state, shape (N, H, T)
        - dWx  : Gradient wrt the input's Weight matrix, shape (D, H)
        - dWh  : Gradient wrt the hidden's Weight matrix, shape (H, H)
        - dbh   : Gradient wrt the bias, shape (1, H)
    '''
    (caches, X) = caches
    (h1, h0, X1, parameters) = caches[0]

    N, H, T = dh.shape
    N, D = X1.shape

    dX = np.zeros((N, D, T))
    dWx = np.zeros((D, H))
    dWh = np.zeros((H, H))
    dbh = np.zeros((1, H))
    dh0 = np.zeros((N, H))
    dh_prevt = np.zeros((N, H))

    for t in reversed(range(T)):
        gradients = rnn_cell_backward(dh[:, :, t] + dh_prevt, caches[t])
        dXt, dh_prevt, dWxt, dWht, dbht = gradients['dXt'], gradients['dh_prev'], gradients['dWx'], gradients['dWh'], gradients['dbh']

        dX[:, :, t] = dXt
        dWx += dWxt
        dWh += dWht
        dbh += dbht

    dh0 = dh_prevt

    gradients = {'dX': dX, 'dh0': dh0, 'dWx': dWx, 'dWh': dWh, 'dbh': dbh}

    return gradients
