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
import matplotlib as mpl
import matplotlib.cm as cmx
import mixture_models
from scipy.misc import factorial
from itertools import product
from itertools import permutations

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
    ex42 = False
    ex43 = True
    ex432 = False
    ex433 = False
    ex44 = False
    ex441 = False
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
        # fig, ax = plt.subplots()
        # for i in np.arange(data2.shape[ 0 ]):
        #     ax.imshow(data2[ i, :, : ].T, cmap='gray_r')
        #     plt.pause(1e-3)
        fig41 = plt.figure()
        ax1 = fig41.add_subplot(131)
        ax1.imshow(data2[ 0, :, : ].T, cmap='gray_r')
        ax2 = fig41.add_subplot(132)
        ax2.imshow(data2[ 1, :, : ].T, cmap='gray_r')
        ax3 = fig41.add_subplot(133)
        ax3.imshow(data2[ 4, :, : ].T, cmap='gray_r')

    """
    Exercise 4.2
    """
    # The class for the BMM algorithm is contained in the mixture models
    if ex42:

        # K = 3 ---------------------------------------------------------
        # set up some stuff for the gaussian mixture models
        n_iterations = 40
        K = 3
        seed = np.random.seed(1)  # shoud be initialised automatically
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
        bmm = bmm.fit(data)
        loglikelihoods = np.sum(bmm.score_samples(data)[ 0 ])
        criterions = bmm.bic(data)
        labels = bmm.score_samples(data)[ 1 ].argmax(axis=1)
        labels += 2

    """
    Exercise 4.3
    """
    if ex43:
        # --------------------------- K = 3 ------------------------------
        K = 3
        n_randomisations = 1
        n_iterations = 45
        for r in np.arange(n_randomisations):
            print('r = {0}'.format(r))
            seed = np.random.seed(r)
            init_means = np.random.random_sample((K, D**2)) * 0.5 + 0.25
            init_weights = np.ones(K, dtype=float) / K
            loglikelihoods = np.zeros((n_iterations, n_randomisations))
            criterions = np.zeros((n_iterations, n_randomisations))
            class_means = np.zeros((n_iterations, n_randomisations, K, D**2))
            labels = np.zeros((data.shape[ 0 ], n_randomisations))
            convergence_print = False
            for i in np.arange(n_iterations):
                print('iteration = {0}'.format(i))
                bmm = mixture_models.MixtureModel(n_components=K,
                                                  means_init=init_means,
                                                  weights_init=init_weights,
                                                  n_iter=i,
                                                  distrib='Bernoulli',
                                                  random_state=seed)
                bmm = bmm.fit(data)
                # give the class means:
                for k in np.arange(K):
                    class_means[ i, r, k, : ] = bmm.means_[ k, : ]
                loglikelihoods[ i, : ] = np.sum(bmm.score_samples(data)[ 0 ])
                criterions[ i, : ] = bmm.bic(data)
                # convergence?
                if bmm.converged_ and not convergence_print:
                    print('converged at step {0}'.format(i))
                    convergence_print = True
                    labels[ :, r ] = bmm.score_samples(data)[ 1 ].argmax(axis=1)
                    labels += 2
                    break
            if not bmm.converged_:
                print('no convergence in trial {0}'.format(r))

    fig431 = plt.figure()
    ax1 = fig431.add_subplot(331)
    ax1.imshow(class_means[ 10, 0, 0, : ].reshape(D, D).T, cmap='gray_r')
    ax2 = fig431.add_subplot(332)
    ax2.imshow(class_means[ 10, 0, 1, : ].reshape(D, D).T, cmap='gray_r')
    ax3 = fig431.add_subplot(333)
    ax3.imshow(class_means[ 10, 0, 2, : ].reshape(D, D).T, cmap='gray_r')
    ax4 = fig431.add_subplot(334)
    ax4.imshow(class_means[ 24, 0, 0, : ].reshape(D, D).T, cmap='gray_r')
    ax5 = fig431.add_subplot(335)
    ax5.imshow(class_means[ 24, 0, 1, : ].reshape(D, D).T, cmap='gray_r')
    ax6 = fig431.add_subplot(336)
    ax6.imshow(class_means[ 24, 0, 2, : ].reshape(D, D).T, cmap='gray_r')
    ax7 = fig431.add_subplot(337)
    ax7.imshow(class_means[ 38, 0, 0, : ].reshape(D, D).T, cmap='gray_r')
    ax8 = fig431.add_subplot(338)
    ax8.imshow(class_means[ 38, 0, 1, : ].reshape(D, D).T, cmap='gray_r')
    ax9 = fig431.add_subplot(339)
    ax9.imshow(class_means[ 38, 0, 2, : ].reshape(D, D).T, cmap='gray_r')

    fig331 = plt.figure()
    ax1 = fig331.add_subplot(121)
    ax1.plot(np.arange(1, n_iterations), loglikelihoods2[ 1:, 0 ], 'b',
             )
    ax1.set_xlabel('Update step of EM-algorithm')
    ax1.set_ylabel('Log-likelihood of sample distribution')
    plt.title(r'$\mathrm{Loglikelihood}$')
    ax2 = fig331.add_subplot(122)
    ax2.plot(np.arange(1, n_iterations), criterions2[ 1:, 0 ], 'b',
             )
    ax2.set_xlabel('Update step of EM-algorithm')
    ax2.set_ylabel('Bayes information criterion')
    plt.title(r'$\mathrm{BIC}$')

    """
    Exercise 4.4
    """
    if ex441:
        # --------------------------- K = 2 ------------------------------
        K = 2
        n_randomisations = 1
        n_iterations = 45
        for r in np.arange(n_randomisations):
            print('r = {0}'.format(r))
            seed = np.random.seed(r)
            init_means = np.random.random_sample((K, D**2)) * 0.5 + 0.25
            init_weights = np.ones(K, dtype=float) / K
            loglikelihoods = np.zeros((n_iterations, n_randomisations))
            criterions = np.zeros((n_iterations, n_randomisations))
            class_means = np.zeros((n_iterations, n_randomisations, K, D**2))
            labels = np.zeros((data.shape[ 0 ], n_randomisations))
            convergence_print = False
            for i in np.arange(n_iterations):
                print('iteration = {0}'.format(i))
                bmm = mixture_models.MixtureModel(n_components=K,
                                                  means_init=init_means,
                                                  weights_init=init_weights,
                                                  n_iter=i,
                                                  distrib='Bernoulli',
                                                  random_state=seed)
                bmm = bmm.fit(data)
                # give the class means:
                for k in np.arange(K):
                    class_means[ i, r, k, : ] = bmm.means_[ k, : ]
                loglikelihoods[ i, : ] = np.sum(bmm.score_samples(data)[ 0 ])
                criterions[ i, : ] = bmm.bic(data)
                # convergence?
                if bmm.converged_ and not convergence_print:
                    print('converged at step {0}'.format(i))
                    convergence_print = True
                    labels[ :, r ] = bmm.score_samples(data)[ 1 ].argmax(
                        axis=1)
                    break
            if not bmm.converged_:
                print('no convergence in trial {0}'.format(r))

        # --------------------------- K = 4 ------------------------------
        K = 4
        n_randomisations = 1
        n_iterations = 45
        for r in np.arange(n_randomisations):
            print('r = {0}'.format(r))
            seed = np.random.seed(r)
            init_means = np.random.random_sample((K, D**2)) * 0.5 + 0.25
            init_weights = np.ones(K, dtype=float) / K
            loglikelihoods = np.zeros((n_iterations, n_randomisations))
            criterions = np.zeros((n_iterations, n_randomisations))
            class_means = np.zeros((n_iterations, n_randomisations, K, D**2))
            labels = np.zeros((data.shape[ 0 ], n_randomisations))
            convergence_print = False
            for i in np.arange(n_iterations):
                print('iteration = {0}'.format(i))
                bmm = mixture_models.MixtureModel(n_components=K,
                                                  means_init=init_means,
                                                  weights_init=init_weights,
                                                  n_iter=i,
                                                  distrib='Bernoulli',
                                                  random_state=seed)
                bmm = bmm.fit(data)
                # give the class means:
                for k in np.arange(K):
                    class_means[ i, r, k, : ] = bmm.means_[ k, : ]
                loglikelihoods[ i, : ] = np.sum(bmm.score_samples(data)[ 0 ])
                criterions[ i, : ] = bmm.bic(data)
                # convergence?
                if bmm.converged_ and not convergence_print:
                    print('converged at step {0}'.format(i))
                    convergence_print = True
                    labels[ :, r ] = bmm.score_samples(data)[ 1 ].argmax(
                        axis=1)
                    break
            if not bmm.converged_:
                print('no convergence in trial {0}'.format(r))

        """
        Point 2:
        """
        performance1 = ground_truth_comparison(labels[ :, 1 ])

        """
        Point 3:
        """
