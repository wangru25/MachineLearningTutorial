import numpy as np

#================================Activation Functions===========================

def sigmoid(input, derivative=False):
	res = 1 / (1 + np.exp(-input))
	if derivative:
		return res * (1 - res)
	return res

def relu(input, derivative=False):
	res = input
	if derivative:
		return 1.0 * (res > 0)
	else:
		return res * (res > 0)
		# return np.maximum(input, 0, input) # ver. 2

def lrelu(input, alpha=0.01, derivative=False):
	res = input
	if derivative:
		dx = np.ones_like(res)
		dx[res < 0] = alpha
		return dx
	else:
		return np.maximum(input, input*alpha, input)

def tanh(input, derivative=False):
	res = np.tanh(input)
	if derivative:
		return 1.0 - np.tanh(input) ** 2
	return res
