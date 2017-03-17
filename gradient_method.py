# coding: utf-8
import numpy as np
import matplotlib.pylab as plt
from common.gradient import numerical_gradient


def function_1(x):
    return x[0]**2 + x[1]**2


def function_2(x):
    return x[0]**2 + 2*x[1]


def gradient_descent(f, init_x, ir=0.001, step_num=100):
    x = init_x.copy()
    x_hist = []

    for i in range(step_num):
        x_hist.append(x.copy())
        grad = numerical_gradient(f, x)
        x -= ir * grad
    x_hist.append(x.copy())

    return x, np.array(x_hist)


if __name__ == '__main__':
    init_x1 = np.array([-3.0, 4.0])
    x1, x_history1 = gradient_descent(function_1, init_x1, ir=0.1, step_num=20)

    plt.plot([-5, 5], [0, 0], '--b')
    plt.plot([0, 0], [-5, 5], '--b')
    plt.plot(x_history1[:, 0], x_history1[:, 1], 'o')

    plt.xlim(-3.5, 3.5)
    plt.ylim(-4.5, 4.5)
    plt.xlabel("X0")
    plt.ylabel("X1")
    plt.show()

    init_x2 = np.array([-3.0, 4.0])
    x2, x_history2 = gradient_descent(function_2, init_x2, ir=0.1, step_num=20)

    plt.plot([-5, 5], [0, 0], '--b')
    plt.plot([0, 0], [-5, 5], '--b')
    plt.plot(x_history2[:, 0], x_history2[:, 1], 'o')

    plt.xlim(-3.5, 3.5)
    plt.ylim(-4.5, 4.5)
    plt.xlabel("X0")
    plt.ylabel("X1")
    plt.show()
