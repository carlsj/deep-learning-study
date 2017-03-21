# coding: utf-8
import numpy as np


class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for key in params.key():
            params[key] -= self.lr * grads[key]


class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            v = {}
            for key in params.key():
                v[key] = np.zeros_like(params[key])

        for key in params.key():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v [key]


class Nesterov:
    """Nesterov’s Accelerated Gradient Descent (http://newsight.tistory.com/224)
        Advanced method of Momentum
    """
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            v = {}
            for key in params.key():
                v[key] = np.zeros_like(params[key])

        # TODO : check formula in theory and write my own term
        for key in params.key():
            """
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]

            self.v[key] = self.v[key] * self.momentum - self.lr * grads[key]
            params[key] = params[key] + self.momentum * self.momentum * self.v[key] \
                          - (1 + self.momentum) * self.lr * grads[key]
            """
            self.v[key] *= self.momentum
            self.v[key] -= self.lr * grads[key]
            params[key] += self.momentum * self.momentum * self.v[key]
            params[key] -= (1 + self.momentum) * self.lr * grads[key]


class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key in params.key():
                self.h[key] = np.zeros_like(params[key])

        for key in params.key():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)


class RMSprop:
    def __init__(self, lr=0.01, dr=0.9):
        self.lr = lr  # learning rate
        self.dr = dr  # decay rate
        self.decay = None

    def update(self, params, grads):
        if self.decay is None:
            self.decay = {}
            for key in params.key():
                self.decay[key] = np.zeros_like(params[key])

        for key in params.key():
            self.decay[key] += self.dr * self.decay[key]
            self.decay[key] = (1 - self.dr) * grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (self.decay[key] + 1e-7)


class Adam:
    def __init__(self, lr=0.001, momentum1=0.9, momentum2=0.999):
        self.lr = lr
        self.momentum1 = momentum1
        self.momentum2 = momentum2
        self.iter = 0
        self.m = None
        self.v = None

    def update(self, params, grads):
        if self.m is None:
            self.m = {}
            self.v = {}
            for key in params.key():
                self.m[key] = np.zeros_like(params[key])
                self.v[key] = np.zeros_like(params[key])

        self.iter += 1

        # Optimized integration
        lr_iter = self.lr * np.sqrt(1.0 - self.momentum2**self.iter) / (1.0 - self.momentum1**self.iter)
        for key in params.key():
            self.m[key] += (1.0 - self.momentum1) * (grads[key] - self.m[key])
            self.v[key] += (1.0 - self.momentum2) * (grads[key]**2 - self.v[key])
            params[key] -= lr_iter * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)

        """ # Original formula integration
        for key in params.key():
            self.m[key] = self.momentum1 * self.m[key] + (1.0 - self.momentum1) * grads[key]
            self.v[key] = self.momentum2 * self.v[key] + (1.0 - self.momentum2) * grads[key]**2

            m_to_apply = self.m[key] / (1.0 - self.momentum1 ** self.iter)
            v_to_apply = self.v[key] / (1.0 - self.momentum2 ** self.iter)
            params[key] -= self.lr * m_to_apply / (np.sqrt(v_to_apply) + 1e-7)
        """