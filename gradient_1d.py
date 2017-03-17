# coding: utf-8
import numpy as np
import matplotlib.pylab as plt


def numerical_gradient(f, x):
    h = 1e-4  # 0.0001
    return (f(x+h) - f(x-h)) / (2*h)


def func1(x):
    return 0.01*(x**2) + 0.1*x


def tangent_line(f, x):
    d = numerical_gradient(f, x)
    y = f(x) - d*x  # To pass f(x) point
    return lambda t: d*t + y


x = np.arange(0.0, 20.0, 0.1)
y = func1(x)
plt.plot(x, y)

tf = tangent_line(func1, 5)
y2 = tf(x)
plt.plot(x, y2)

plt.xlabel('x')
plt.ylabel('f(x)')
plt.show()