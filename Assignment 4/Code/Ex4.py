# coding=utf-8
"""
This file contains solutions to Exercise 4 of Assignment 4 of Bert
 Kappen's course "Statistical Machine Learning" 2016/2017.
"""

import numpy as np
from scipy import linalg
import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from time import time
import numbers
from sklearn import cluster
import matplotlib as mpl
import matplotlib.cm as cmx

cm = mpl.colors.ListedColormap('YlGnBu')
seashore = cm = plt.get_cmap('YlGnBu')
scalarMap = cmx.ScalarMappable(cmap=seashore)

if __name__ == '__main__':
    """
    Exercise 4.1
    """

    # load the image data
    N = 800
    D = 28
    X = np.zeros((N, D), dtype=int)

    data = np.fromfile('../Data/a012_images.dat', dtype=np.int8)
    data = data.reshape(N, D, D)
    data = np.array(data, dtype=int)

    fig, ax = plt.subplots()
    for i in range(data.shape[ 0 ]):
        ax.imshow(data[ i, :, : ].T)
        pause(1e-3)

