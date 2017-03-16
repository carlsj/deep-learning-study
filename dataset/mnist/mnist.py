# coding: utf-8
# ref: https://github.com/WegraLee/deep-learning-from-scratch


try:
    import urllib.request
except ImportError:
    raise ImportError('You need Python 3.x')
import os
import os.path
import gzip
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


def _load_label(file_name):
    file_path = dataset_dir + "/" + file_name

    # TODO: return num_labels
    print("Converting" + file_name + "to NumPy Arrary...")
    with gzip.open(file_path, 'rb') as f:
        dt = np.dtype(int)
        dt = dt.newbyteorder('>')
        num_labels = int(np.frombuffer(f.read(8), dt, 1, offset=4))  # offset to skip magic number
        labels = np.frombuffer(f.read(), np.uint8)
    print("Done")
    return labels


def _load_img(file_name):
    file_path = dataset_dir + "/" + file_name

    # TODO: return num_imgs
    print("Converting" + file_name + "to NumPy Arrary...")
    with gzip.open(file_path, 'rb') as f:
        dt = np.dtype(int)
        dt = dt.newbyteorder('>')
        num_imgs = int(np.frombuffer(f.read(8), dt, 1, offset=4))  # offset to skip magic number
        num_rows = int(np.frombuffer(f.read(4), dt, 1))
        num_cols = int(np.frombuffer(f.read(4), dt, 1))
        img_data = np.frombuffer(f.read(), np.uint8)
    img_data = img_data.reshape(-1, num_rows*num_cols)
    print("Done")
    return img_data


def _get_dataset_numpy():
    dataset = {}
    dataset['train_img'] = _load_img(dataset_file['train_img'])
    dataset['train_label'] = _load_img(dataset_file['train_label'])
    dataset['test_img'] = _load_img(dataset_file['test_img'])
    dataset['test_label'] = _load_img(dataset_file['test_label'])


if __name__ == "__main__":
    download_mnist()
    dataset = {}
    dataset['test_label'] = _load_label(dataset_file['test_label'])
    dataset['test_img'] = _load_img(dataset_file['test_img'])
    pass

