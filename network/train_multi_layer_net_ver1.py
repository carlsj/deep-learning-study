# coding: utf-8
import matplotlib.pylab as plt
from dataset.mnist.mnist import load_mnist
from network.multi_layer_net_ver1 import *

sys.path.append(os.pardir)


# Load MNIST data
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=True)

# Hyper Parameters
train_size = x_train.shape[0]  # 60000
input_size = x_train.shape[1]  # 784
hidden_size_list = [100, 100]
output_size = 10
weight_init_std = 0.01
iters_num = 100000
lr = 0.1
batch_size = 100

activation='relu'
weight_init_std='relu'
weight_decay_lamda = 0

# Create Network
mnist_nn = MultiLayerNetVer1(input_size, hidden_size_list, output_size,
                             activation, weight_init_std, weight_decay_lamda)
iter_per_epoch = max(train_size / batch_size, 1)

train_loss_list = []
train_acc_list = []
test_acc_list = []

# Training
for ind_iters in range(0, iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # Compute gradient
    grads = mnist_nn.gradient(x_batch, t_batch)

    # Update parameters
    for key in ('W1', 'b1', 'W2', 'b2'):
        mnist_nn.params[key] -= lr * grads[key]

    # Record loss
    loss = mnist_nn.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    # Accuracy in every epoch
    if ind_iters % iter_per_epoch == 0:
        train_acc = mnist_nn.accuracy(x_train, t_train)
        train_acc_list.append(train_acc)
        test_acc = mnist_nn.accuracy(x_test, t_test)
        test_acc_list.append(test_acc)
        print("train acc / test acc : " + str(train_acc) + "/" + str(test_acc))

# Plot
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train_acc')
plt.plot(x, test_acc_list, label='test_acc', linestyle='--')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()

