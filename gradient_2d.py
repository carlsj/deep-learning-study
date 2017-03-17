# coding: utf-8
# cf.http://d.hatena.ne.jp/white_wheels/20100327/p3
import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D


def numerical_gradient_2d(f, x, y):
    h = 1e-4
    grad_x = (f(x+h, y) - f(x-h, y)) / 2*h
    grad_y = (f(x, y+h) - f(x, y-h)) / 2*h
    return grad_x, grad_y


def function_2(x, y):
    return x**2 + y**2


def function_3(x, y):
    return x**2 + 2*y


if __name__ == '__main__':
    x0 = np.arange(-2, 2.5, 0.25)
    x1 = np.arange(-2, 2.5, 0.25)
    X, Y = np.meshgrid(x0, x1)

    # Display function_2
    Z = function_2(X, Y)

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_wireframe(X, Y, Z)
    plt.show()

    # Display function_3
    Z = function_3(X, Y)

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(X, Y, Z)
    plt.show()

    X = X.flatten()
    Y = Y.flatten()

    # 2d gradient of function_2
    grad_x, grad_y = numerical_gradient_2d(function_2, X, Y)
    plt.figure()
    plt.quiver(X, Y, -grad_x, -grad_y, angles="xy", color="#666666")  # ,headwidth=10,scale=40,color="#444444")
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.grid()
    plt.legend()
    plt.draw()
    plt.show()

    # 2d gradient of function_3
    grad_x, grad_y = numerical_gradient_2d(function_3, X, Y)
    plt.figure()
    plt.quiver(X, Y, -grad_x, -grad_y, angles="xy", color="#444444")  # ,headwidth=10,scale=40,color="#444444")
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.grid()
    plt.legend()
    plt.draw()
    plt.show()
