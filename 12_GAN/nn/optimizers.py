import numpy as np


def _copy_weights_to_zeros(weights):
    result = {}
    result.keys()
    for key in weights.keys():
        result[key] = np.zeros_like(weights[key])
    return result


class SGD(object):


    def __init__(self, weights, lr=0.01, momentum=0.9, decay=1e-5):

        self.v = _copy_weights_to_zeros(weights)
        self.iterations = 0
        self.lr = self.init_lr = lr
        self.momentum = momentum
        self.decay = decay

    def iterate(self, weights, gradients):

        self.lr = self.init_lr / (1 + self.iterations * self.decay)

        for key in self.v.keys():
            self.v[key] = self.momentum * self.v[key] + self.lr * gradients[key]
            weights[key] = weights[key] - self.v[key]

        self.iterations += 1
