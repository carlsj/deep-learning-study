# coding: utf-8
import numpy as np


def _numerical_gradient_1d(f, x):
    """
    :param f: Multi variable function.
               e.g. f(y) = x1**2 + 2*x2 + x*3
    :param x: variables of function 'f'.
               e.g. (x1, x2 ...)
    :return: Gradient of the function 'f' when 'x' is input
              Size is same as 'x'
    """
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)

    # Run through each variables of the function
    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)

        x[idx] = float(tmp_val - h)
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val

    return grad


def numerical_gradient_2d(f, X):
    """
    :param f: Multi variable function.
               e.g. f(y) = x1**2 + 2*x2 + x*3
    :param X: (n,  (x1, x2 ...))
    :return: Gradient of the function 'f' when 'X' is input
              Size is same as X
    """
    if X.ndim == 1:
        return _numerical_gradient_1d(f, X)
    else:
        grad = np.zeros_like(X)

        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient_1d(f, x)

        return grad


def numerical_gradient(f, x):
    # TODO : Complete comment
    """
    :param f: ?
    :param x: ?
    :return: Gradient of the function 'f' when 'x' is input
              Size is same as x
    """
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)

    iter_ = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not iter_.finished:
        idx = iter_.multi_index
        tmp_val = x[idx]

        x[idx] = float(tmp_val) + h
        fxh1 = f(x)

        x[idx] = float(tmp_val) - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val
        iter_.iternext()

    return grad


