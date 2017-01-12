# coding=utf-8
"""
This file contains solutions to Exercise 4 of Assignment 4 of Bert
 Kappen's course "Statistical Machine Learning" 2016/2017.
"""

import numpy as np
from scipy import linalg
import os
import copy
import matplotlib.pyplot as plt
import numbersimport mixture_models
from scipy.misc import factorial
from itertools import product
from itertools import permutations

# cm = mpl.colors.ListedColormap('YlGnBu')
# seashore = cm = plt.get_cmap('YlGnBu')
# scalarMap = cmx.ScalarMappable(cmap=seashore)

def ground_truth_comparison(labels):
    """Take classification and compare to MNIST labels"""
    # now let's compare to the ground truth:
    ground_truth = np.fromfile('../Data/a012_labels.dat', dtype=np.int8)
    ground_truth = np.array(ground_truth, dtype=int)

    # we need to permute the labels, to see how many numbers were
    # identified correctly!
    permutation_list = [ ]
    agreement = np.zeros(int(factorial(len(np.unique(labels)))))
    for i, p in enumerate(permutations(np.unique(labels))):
        relabeled = np.zeros_like(labels)
        permutation_list.append(p)
        for j in np.arange(len(np.unique(labels))):
            relabeled[ labels == np.unique(labels)[ j ] ] = p[ j ]
        agreement[ i ] = np.sum(relabeled == ground_truth)
    relabeled = np.zeros_like(labels)
    for j in np.arange(len(np.unique(labels))):
        relabeled[ labels == np.unique(labels)[ j ] ] = permutation_list[
            np.argmax(agreement) ][ j ]

    # how well did we perform?
    performance = np.sum(relabeled == ground_truth)
    return performance


if __name__ == '__main__':

    ex41 = False  # False just supresses the output, but still loads the data
    ex42 = True
    ex43 = True
    ex44 = True
    ex45 = False

    """
    Exercise 4.1
    """
    # load the image data
    N = 800
    D = 28

    data = np.fromfile('../Data/a012_images.dat', dtype=np.int8)
    data = np.array(data, dtype=int)
    data2 = data.reshape(N, D, D)  # just for visualisation
    data = np.array(data.reshape(N, D**2), dtype=float)

    if ex41:
        fig, ax = plt.subplots()
        for i in np.arange(data2.shape[ 0 ]):
            ax.imshow(data2[ i, :, : ].T, cmap='gray_r')
            plt.pause(1e-3)

    """
    Exercise 4.2
    """
    # The class for the BMM algorithm is contained in the mixture models
    if ex42:
        # let's first reshape the data to array format

        # set up some stuff for the gaussian mixture models
        n_iterations = 40
        K = 3
#        np.random.seed(1)  # shoud be initialised automatically
        # initialisation
        init_means = np.random.random_sample((K, D**2)) * 0.5 + 0.25
        init_weights = np.ones(K, dtype=float) / K

        # Fit a Gaussian mixture with EM using five components
        loglikelihoods = np.zeros(n_iterations)
        criterions = np.zeros(n_iterations)
        convergence_print = False

        # let's initialise the model for a single run.
        bmm = mixture_models.MixtureModel(n_components=K,
                              means_init=init_means,
                              weights_init=init_weights,
                              n_iter=n_iterations,
                              distrib='Bernoulli')
    """
    Exercise 4.3
    """
    if ex43:
        bmm = bmm.fit(data)
        # this leads to convergence:
        bmm.converged_
        # let's get our labels
        labels = bmm.score_samples(data)[ 1 ].argmax(axis=1)
        labels += 2

    """
    Exercise 4.4
    """
    if ex44:
        """
        Point 1:
        """

        """
        Point 2:
        """
        performance1 = ground_truth_comparison(labels)

        """
        Point 3:
        """
