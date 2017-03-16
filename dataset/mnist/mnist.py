# coding: utf-8
# ref: https://github.com/WegraLee/deep-learning-from-scratch


try:
    import urllib.request
except ImportError:
    raise ImportError('You need Python 3.x')
import os
import os.path
import gzip
import pickle
import numpy as np


url_base = 'http://yann.lecun.com/exdb/mnist/'
dataset_file = {
    'train_img':'train-images-idx3-ubyte.gz',
    'train_label':'train-labels-idx1-ubyte.gz',
    'test_img':'t10k-images-idx3-ubyte.gz',
    'test_label':'t10k-labels-idx1-ubyte.gz'
}

dataset_dir = os.path.dirname(os.path.abspath(__file__))
save_file = dataset_dir + "/mnist.pkl"

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
    dataset['train_label'] = _load_label(dataset_file['train_label'])
    dataset['test_img'] = _load_img(dataset_file['test_img'])
    dataset['test_label'] = _load_label(dataset_file['test_label'])
    return dataset

def init_mnist():
    download_mnist()
    dataset = _get_dataset_numpy()
    print("Creating pickle file...")
    with open(save_file, 'wb') as f:
        pickle.dump(dataset, f, -1)
    print("Done!")
    return dataset


def _change_one_hot_encoding(x):
    one_hot_encoding = np.zeros((x.size, 10))
    for idx, row in enumerate(one_hot_encoding):
        row[x[idx]] = 1
    return one_hot_encoding


def load_mnist(normalize=False, flatten=False, one_hot_label=False):

    if not os.path.exists(save_file):
        dataset = init_mnist()
    else:
        with open(save_file, 'rb') as f:
            dataset = pickle.load(f)

    if normalize:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].astype(np.float32) / 255.0
        pass

    if not flatten:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)
        pass

    if one_hot_label:
        dataset['train_label'] = _change_one_hot_encoding(dataset['train_label'])
        dataset['test_label'] = _change_one_hot_encoding(dataset['test_label'])

    return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label'])


if __name__ == "__main__":
    (train_img, train_label), (test_img, test_label) = load_mnist()
    (train_img, train_label), (test_img, test_label) = load_mnist(normalize=True, flatten=True, one_hot_label=True)
    pass

