# coding: utf-8
import sys
import os
from collections import OrderedDict
from common.layers import *

sys.path.append(os.pardir)


class MultiLayerNetVer1:

    def __init__(self, input_size, hidden_size_list, output_size,
                 activation='relu', weight_init_std='relu', weight_decay_lamda=0):
        """Fully Connected Multi-Layer Neural Network

        :param input_size: Size of input (e.g. MNIST : 784)
        :param hidden_size_list: A list of hidden layers (e.g. [100, 100, 100])
        :param output_size: Size of output (e.g. MNIST : 10)
        :param activation: Activation Function ('relu' or 'sigmoid')
        :param weight_init_std: Stand Deviation to initialize weights (e.g. 0.01)
                                 In case 'relu' or 'he', it would be 'He' initialize value
                                 In case 'sigmoid' or 'xavier', it would be 'Xavier' initialize value
        :param weight_decay_lamda: A strength of weight decay according to 'L2' Law
        """
        # Initialize parameters
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list)
        self.weight_decay_lamda = weight_decay_lamda
        self.params = {}

        # Initialize weights
        self.__init_weight(weight_init_std)

        # Create layers
        # Number of total layer is (hidden_layer_num + 1) * 2
        activation_layer = {'sigmoid': Sigmoid, 'relu': Relu}
        self.layers = OrderedDict()
        for idx in range(1, self.hidden_layer_num+1):
            self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)],
                                                      self.params['b' + str(idx)])
            self.layers['ActFunc' + str(idx)] = activation_layer[activation]()

        idx = self.hidden_layer_num + 1
        self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)],
                                                  self.params['b' + str(idx)])

        self.last_layer = SoftmaxWithLoss()

    def __init_weight(self, weight_init_std):
        """Initialize weights
        :param weight_init_std: Stand Deviation to initialize weights (e.g. 0.01)
                         In case 'relu' or 'he', it would be 'He' initialize value
                         In case 'sigmoid' or 'xavier', it would be 'Xavier' initialize value
        """
        all_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]
        for idx in range (1, len(all_size_list)):
            if str(weight_init_std).lower() in ('relu', 'he'):
                scale = np.sqrt(2.0 / all_size_list[idx - 1])
            elif str(weight_init_std).lower() in ('sidmoid', 'xavier'):
                scale = np.sqrt(1.0 / all_size_list[idx - 1])
            else:
                scale = weight_init_std  # initial scale
            self.params['W' + str(idx)] = scale * np.random.rand(all_size_list[idx - 1], all_size_list[idx])
            self.params['b' + str(idx)] = np.zeros(all_size_list[idx])

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)

        # Compute weight decay
        weight_decay = 0
        for idx in range(1, self.hidden_layer_num + 2):
            W = self.params['W' + str(idx)]
            # TODO : check reason using np.sum()
            weight_decay += 0.5 * self.weight_decay_lamda * np.sum(W**2)

        return self.last_layer.forward(y, t) + weight_decay

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:  # When t is in one-hot-encoding
            t = np.argmax(t, axis=1)
        else:
            pass

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def gradient(self, x, t):
        """Compute gradient
        :param x: Input data
        :param t: Ground truth label
        :return: Dictionary variables of weight and bias
                  grads['W1'], grads['W2'], ... weights
                  grads['b1'], grads['b2'], ... bias
        """
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        for idx in range(1, self.hidden_layer_num + 2):
            grads['W' + str(idx)] = self.layers['Affine' + str(idx)].dW \
                                    + self.weight_decay_lamda * self.layers['Affine' + str(idx)].W
            grads['b' + str(idx)] = self.layers['Affine' + str(idx)].db

        return grads








