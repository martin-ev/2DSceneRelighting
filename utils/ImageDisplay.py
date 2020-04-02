#from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

import matplotlib.pyplot as plt
import numpy as np
import torchvision as tv

def imshow(img):
    try:
        npimg = img.numpy()
    except:
        npimg = img
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    
def show_some(dataloader):
    # get some random training images
    dataiter = iter(dataloader)
    images, _ = dataiter.next()

    # show images
    imshow(tv.utils.make_grid(images))