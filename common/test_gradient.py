# coding: utf-8
import matplotlib.pylab as plt
from common.gradient import *


def function_1(x):
    return x[0]**2 + x[1]**2


def function_2(x):
    return x[0]**2 + 2*x[1]


if __name__ == '__main__':

    x0 = np.arange(-2, 2.5, 0.25)
    x1 = np.arange(-2, 2.5, 0.25)
    X, Y = np.meshgrid(x0, x1)

    X = X.flatten()
    Y = Y.flatten()

    # 2d gradient of function_1
    grad = numerical_gradient_2d(function_1, np.array([X, Y]).T)
    grad = grad.T

    plt.figure()
    plt.quiver(X, Y, -grad[0], -grad[1], angles="xy", color="#666666")
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.grid()
    plt.legend()
    plt.draw()
    plt.show()

    # 2d gradient of function_2
    grad = numerical_gradient_2d(function_2, np.array([X, Y]).T)
    grad = grad.T
    plt.figure()
    plt.quiver(X, Y, -grad[0], -grad[1], angles="xy", color="#666666")
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.grid()
    plt.legend()
    plt.draw()
    plt.show()

    # TODO : Test for numerical_gradient
    # a. When 'x' is variable of 'f'
    # b. When 'x' is constant variable of 'f'
    """
    tmp_x = np.array([0.1, 0.2])
    grad_3 = numerical_gradient(function_1, tmp_x)
    print(grad_3)
    """
