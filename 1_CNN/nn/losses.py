import numpy as np
import math

def cross_entropy_loss(y_pred, y):
    """
    Cross entropy loss
    y_pred: predict y, shape:(M,d), M is the #of samples
    y: shape(M,d)
    """
    y_shift = y_pred - np.max(y_pred, axis=-1, keepdims=True)
    y_exp = np.exp(y_shift)
    y_hat = y_exp / np.sum(y_exp, axis=-1,keepdims=True)
    loss = np.mean(np.sum(-y * np.log(y_hat), axis=-1))
    dy = y_hat - y
    return loss, dy
