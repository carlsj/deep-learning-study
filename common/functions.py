# coding: utf-8
import numpy as np


def identity_function(x):
    return x


def step_function(x):
    return np.array(x > 0, dtype=np.int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_grad(x):
    return sigmoid(x) * (1.0 - sigmoid(x))


def relu(x):
    return np.maximum(0, x)


def softmax(x):
    if x.ndim == 1:
        x_sift = x - np.max(x)  # To prevent overflow
        return np.exp(x_sift) / sum(np.exp(x_sift))

    elif x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    else:
        pass


def mean_square_err(y, t):
    return 0.5 * np.sum((y - t)**2)


def cross_entropy_err(y, t):
    if y.ndim == 1:  # For consistency with batch mode
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)

    if t.size == y.size:  # When label is one-hot-encoding
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t]))


def softmax_loss(X, t):
    y = softmax(X)
    return cross_entropy_err(y, t)


