# coding: utf-8
# ref: https://github.com/WegraLee/deep-learning-from-scratch

try:
    import urllib.request
except ImportError:
    raise ImportError('You need Python 3.x')
import os.path
import gzip
import os
import numpy as np


url_base = 'http://yann.lecun.com/exdb/mnist/'
dataset_file = {
    'train_img':'train-images-idx3-ubyte.gz',
    'train_label':'train-labels-idx1-ubyte.gz',
    'test_img':'t10k-images-idx3-ubyte.gz',
    'test_label':'t10k-labels-idx1-ubyte.gz'
}

dataset_dir = os.path.dirname(os.path.abspath(__file__))

train_num = 60000
test_num = 10000
img_dim = (1, 28, 28)
img_size = img_dim[0] * img_dim[1] * img_dim[2]


def _download(file_name):
    file_path = dataset_dir + "/" + file_name

    if os.path.exists(file_path):
        print("The file: " + file_name + "exists already")
    else:
        print("Downloading " + file_name + "...")
        urllib.request.urlretrieve(url_base + file_name, file_path)


def download_mnist():
    for val in dataset_file.values():
        _download(val)

if __name__ == "__main__":
    download_mnist()
    pass

