# From https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

import matplotlib.pyplot as plt
import numpy as np
import torchvision as tv


def imshow(img):
    try:
        numpy_img = img.numpy()
    except:
        numpy_img = img
    plt.imshow(np.transpose(numpy_img, (1, 2, 0)))
    plt.show()


def show_some(dataloader):
    # get some random training images
    data_iter = iter(dataloader)
    images, _ = data_iter.next()

    # show images
    imshow(tv.utils.make_grid(images))
