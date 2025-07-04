import os
import numpy as np


def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)


def calculate_scalar(x):
    print(x.shape)
    if x.ndim == 2:
        axis = 0
    elif x.ndim == 3:
        axis = (0, 1)

    mean = np.mean(x, axis=axis)
    std = np.std(x, axis=axis)
    return mean, std


def scale(x, mean, std):
    return (x - mean) / std








