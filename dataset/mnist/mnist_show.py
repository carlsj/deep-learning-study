# coding: utf-8
import sys
import os
from math import sqrt
from math import ceil
import numpy as np
from PIL import Image
from dataset.mnist.mnist import load_mnist


sys.path.append(os.pardir)  # to access parent dir


def img_show(img, num_img_show=100):
    grid_size = ceil(sqrt(num_img_show))
    pil_img_size = 28 * grid_size
    pil_img_show = Image.new(mode='P', size=(pil_img_size, pil_img_size), color=255)

    count_col = 0
    count_row = 0
    for idx in range(0, num_img_show):
        if idx > train_img.shape[0]:
            break;
        else:
            pass

        img_tmp = train_img[idx]
        img_tmp = img_tmp.reshape(28, 28)
        pil_img_tmp = Image.fromarray(np.uint8(img_tmp))

        offset = (28*count_col, 28*count_row)
        pil_img_show.paste(pil_img_tmp, offset)

        if count_col < grid_size-1:
            count_col += 1
        else:
            count_col = 0
            count_row += 1

    pil_img_show.show()


if __name__ == '__main__':
    (train_img, train_label), (test_img, test_label) = load_mnist()
    img_show(train_img, num_img_show=100)



