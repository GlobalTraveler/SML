"""
This is a default script that should be adapted to the respective purpose.
"""

import numpy as np
from scipy import linalg
import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numbers
import matplotlib as mpl
import matplotlib.cm as cmx
import mixture_models
from itertools import combinations

cm = mpl.colors.ListedColormap('YlGnBu')
seashore = cm = plt.get_cmap('YlGnBu')
scalarMap = cmx.ScalarMappable(cmap=seashore)
plt.clf()

if __name__ == '__main__':
    # let us first load the data:
    os.chdir('../Data/')
    X = np.loadtxt('a011_mixdata.txt')
    os.chdir('../Code/')
    N, D = X.shape
    # n = number of datapoints
    # D = number of features

    ex31 = False
    ex32 = False
    ex33 = False
    ex34 = False
    ex35 = True

    if ex31:
        """
        Exercise 3.1
        """
        fig = plt.figure()
        ax = fig.add_subplot(121, projection='3d')
        ax.scatter(X[ :, 0 ], X[ :, 1 ], X[ :, 2 ], c='c', marker='o',  s=6, alpha=0.75)
        ax.set_xlabel('1st variable')
        ax.set_ylabel('2nd variable')
        ax.set_zlabel('3rd variable')
        plt.title(r'$\mathrm{Point\ cloud\ in\ dims\ 1-3}$')
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.scatter(X[ :, 0 ], X[ :, 2 ], X[ :, 3 ], c='c', marker='o',  s=6, alpha=0.75)
        ax2.set_xlabel('1st variable')
        ax2.set_ylabel('2th variable')
        ax2.set_zlabel('4th variable')
        plt.title(r'$\mathrm{Point\ cloud\ in\ dims\ 1,\ 2,\ 4}$')

        fig31 = plt.figure()
        ax1 = fig31.add_subplot(221)
        n1, bins1, patches1 = plt.hist(X[ :, 0 ], 25, normed=1, alpha=0.75)
        plt.xlabel('frequency (normed)')
        plt.ylabel('value')
        plt.title(r'$\mathrm{Histogram\ in\ dim\ 1}$')
        ax2 = fig31.add_subplot(222)
        n2, bins2, patches2 = plt.hist(X[ :, 1 ], 25, normed=1, alpha=0.75)
        plt.xlabel('frequency (normed)')
        plt.ylabel('value')
        plt.title(r'$\mathrm{Histogram\ in\ dim\ 2}$')
        ax3 = fig31.add_subplot(223)
        n3, bins3, patches3 = plt.hist(X[ :, 2 ], 25, normed=1, alpha=0.75)
        plt.xlabel('frequency (normed)')
        plt.ylabel('value')
        plt.title(r'$\mathrm{Histogram\ in\ dim\ 3}$')
        ax4 = fig31.add_subplot(224)
        n4, bins4, patches4 = plt.hist(X[ :, 3 ], 25, normed=1, alpha=0.75)
        plt.xlabel('frequency (normed)')
        plt.ylabel('value')
        plt.title(r'$\mathrm{Histogram\ in\ dim\ 4}$')
        plt.show()

    if ex32:
        """
        Exercise 3.2
        """
        # set up some stuff for the gaussian mixture models
        n_iterations = 100
        K = 2
        np.random.seed(1)
        # initialisation
        init_means = np.repeat(np.mean(X, axis=0), K, axis=0).reshape(K, D)
        init_means += np.random.random_sample((K, D)) * 2.0 - 1.0
        init_weights = np.ones(K, dtype=float) / K
        init_covars = np.zeros((K, D, D))
        for k in np.arange(K):
            # the initialisation cannot lead to singularity.
            Sigma_k = np.random.random_sample(D) * 4.0 + 2.0
            init_covars[ k, :, : ] = np.diag(Sigma_k)
        init_covars = init_covars[ :, :, : ]

        # Fit a Gaussian mixture with EM using five components
        loglikelihoods = np.zeros(n_iterations)
        criterions = np.zeros(n_iterations)
        convergence_print = False
        for i in np.arange(1, n_iterations):
            gmm = mixture_models.MixtureModel(
                    n_components=K,
                    means_init=init_means,
                    weights_init=init_weights,
                    covars_init=init_covars,
                    random_state=np.random.seed(1),
                    n_iter=i)
            gmm = gmm.fit(X)
            if gmm.converged_ and not convergence_print:
                print('converged at step {0}'.format(i))
                convergence_print = True
            loglikelihoods[ i ] = np.sum(gmm.score_samples(X)[ 0 ])
            criterions[ i ] = gmm.bic(X)

        # The final data labels:
        labels = gmm.score_samples(X)[ 1 ].argmax(axis=1)

        # let's plot the change in the loglikelihood over iterations
        fig321 = plt.figure()
        ax1 = fig321.add_subplot(121)
        ax1.plot(np.arange(1, n_iterations), loglikelihoods[ 1: ])
        plt.axvline(x=32, color='g')
        ax1.set_xlabel('Update step of EM-algorithm')
        ax1.set_ylabel('Log-likelihood of sample distribution')
        plt.title(r'$\mathrm{Loglikelihood}$')
        ax2 = fig321.add_subplot(122)
        ax2.plot(np.arange(1, n_iterations), criterions[ 1: ])
        plt.axvline(x=32, color='g')
        ax2.set_xlabel('Update step of EM-algorithm')
        ax2.set_ylabel('Bayes information criterion')
        plt.title(r'$\mathrm{BIC}$')

        # Now plot the requested colour-coded first two variables
        fig322 = plt.figure()
        ax3 = fig322.add_subplot(111)
        set0 = X[ labels == 0, : ]
        set1 = X[ labels == 1, : ]
        ax3.scatter(set0[ :, 0 ], set0[ :, 1 ], color='b', s=6, alpha=0.75)
        ax3.scatter(set1[ :, 0 ], set1[ :, 1 ], color='r', s=6, alpha=0.75)
        ax3.set_xlabel(r'$x_1$-dimension')
        ax3.set_ylabel(r'$x_2$-dimension')
        plt.title(r'$\mathrm{BIC}$')
        plt.show()

    if ex33:
        """
        Exercise 3.3
        """
        # The above version was a test run. Now some different initialisations:
        randomisations = 8
        n_iterations = 50
        loglikelihoods2 = np.zeros((n_iterations, randomisations))
        criterions2 = np.zeros((n_iterations, randomisations))
        labels2 = np.zeros((X.shape[ 0 ], randomisations))
        K = 2

        for j in np.arange(randomisations):
            np.random.seed(j)

            # initialisation
            init_means = np.repeat(np.mean(X, axis=0), K, axis=0).reshape(K, D)
            init_means += np.random.random_sample((K, D)) * 2.0 - 1.0
            init_weights = np.ones(K, dtype=float) / K
            init_covars = np.zeros((K, D, D))
            for k in np.arange(K):
                # the initialisation cannot lead to singularity.
                Sigma_k = np.random.random_sample(D) * 4.0 + 2.0
                init_covars[ k, :, : ] = np.diag(Sigma_k)
            init_covars = init_covars[ :, :, : ]

            # Fit a Gaussian mixture with EM using five components
            convergence_print = False
            for i in np.arange(0, n_iterations):
                # the data get quite big so we overwrite it every time
                gmm2 = mixture_models.MixtureModel(
                        n_components=K, n_iter=i,
                        means_init=init_means, weights_init=init_weights,
                        covars_init=init_covars,
                        random_state=np.random.seed(randomisations)
                )
                gmm2 = gmm2.fit(X)
                if gmm2.converged_ and not convergence_print:
                    print('converged at step {0}'.format(i))
                    convergence_print = True
                loglikelihoods2[ i, : ] = np.sum(gmm2.score_samples(X)[ 0 ])
                criterions2[ i, : ] = gmm2.bic(X)
            if not gmm2.converged_:
                print('no convergence in trial {0}'.format(j))
            # The final data labels:
            labels2[ :, j ] = gmm2.score_samples(X)[ 1 ].argmax(axis=1)

        # let's plot the change in the loglikelihood over iterations
        fig331 = plt.figure()
        ax1 = fig331.add_subplot(121)
        ax1.plot(np.arange(1, n_iterations), loglikelihoods2[ 1:, 1 ], 'r',
                 np.arange(1, n_iterations), loglikelihoods2[ 1:, 2 ], 'g',
                 np.arange(1, n_iterations), loglikelihoods2[ 1:, 3 ], 'b',
                 np.arange(1, n_iterations), loglikelihoods2[ 1:, 4 ], 'y',
                 np.arange(1, n_iterations), loglikelihoods2[ 1:, 5 ], 'r--',
                 np.arange(1, n_iterations), loglikelihoods2[ 1:, 6 ], 'g--',
                 np.arange(1, n_iterations), loglikelihoods2[ 1:, 7 ], 'b--',
                 np.arange(1, n_iterations), loglikelihoods2[ 1:, 0 ], 'y--'
                 )
        ax1.set_xlabel('Update step of EM-algorithm')
        ax1.set_ylabel('Log-likelihood of sample distribution')
        plt.title(r'$\mathrm{Loglikelihood}$')
        ax2 = fig331.add_subplot(122)
        ax2.plot(np.arange(1, n_iterations), criterions2[ 1:, 1 ], 'r',
                 np.arange(1, n_iterations), criterions2[ 1:, 2 ], 'g',
                 np.arange(1, n_iterations), criterions2[ 1:, 3 ], 'b',
                 np.arange(1, n_iterations), criterions2[ 1:, 4 ], 'y',
                 np.arange(1, n_iterations), criterions2[ 1:, 5 ], 'r--',
                 np.arange(1, n_iterations), criterions2[ 1:, 6 ], 'g--',
                 np.arange(1, n_iterations), criterions2[ 1:, 7 ], 'b--',
                 np.arange(1, n_iterations), criterions2[ 1:, 0 ], 'y--'
                 )
        ax2.set_xlabel('Update step of EM-algorithm')
        ax2.set_ylabel('Bayes information criterion')
        plt.title(r'$\mathrm{BIC}$')

        # scatter plots
        fig332 = plt.figure()
        ax0 = fig332.add_subplot(221)
        set0 = X[ labels2[ :, 0 ] == 0, : ]
        set1 = X[ labels2[ :, 0 ] == 1, : ]
        ax0.scatter(set0[ :, 0 ], set0[ :, 1 ], color='b', s=6, alpha=0.75)
        ax0.scatter(set1[ :, 0 ], set1[ :, 1 ], color='r', s=6, alpha=0.75)
        ax0.set_xlabel(r'$x_1$-dimension')
        ax0.set_ylabel(r'$x_2$-dimension')
        ax1 = fig332.add_subplot(222)
        set0 = X[ labels2[ :, 2 ] == 0, : ]
        set1 = X[ labels2[ :, 2 ] == 1, : ]
        ax1.scatter(set0[ :, 0 ], set0[ :, 1 ], color='b', s=6, alpha=0.75)
        ax1.scatter(set1[ :, 0 ], set1[ :, 1 ], color='r', s=6, alpha=0.75)
        ax1.set_xlabel(r'$x_1$-dimension')
        ax1.set_ylabel(r'$x_2$-dimension')
        ax2 = fig332.add_subplot(223)
        set0 = X[ labels2[ :, 4 ] == 0, : ]
        set1 = X[ labels2[ :, 4 ] == 1, : ]
        ax2.scatter(set0[ :, 0 ], set0[ :, 1 ], color='b', s=6, alpha=0.75)
        ax2.scatter(set1[ :, 0 ], set1[ :, 1 ], color='r', s=6, alpha=0.75)
        ax2.set_xlabel(r'$x_1$-dimension')
        ax2.set_ylabel(r'$x_2$-dimension')
        ax3 = fig332.add_subplot(224)
        set0 = X[ labels2[ :, 6 ] == 0, : ]
        set1 = X[ labels2[ :, 6 ] == 1, : ]
        ax3.scatter(set0[ :, 0 ], set0[ :, 1 ], color='b', s=6, alpha=0.75)
        ax3.scatter(set1[ :, 0 ], set1[ :, 1 ], color='r', s=6, alpha=0.75)
        ax3.set_xlabel(r'$x_1$-dimension')
        ax3.set_ylabel(r'$x_2$-dimension')

        # correlation coefficients
        corrcoeffs2 = np.zeros((K, randomisations))
        for l in np.arange(randomisations):
            for k in np.arange(K):
                submat = X[ labels2[ :, l ] == k, : ]
                corrcoeffs2[ k, l ] = \
                    np.corrcoef(submat[ :, 0 ], submat[ :, 1 ])[ 0, 1 ]
        corrcoeffs_mean2 = np.nanmean(corrcoeffs2, axis=1)
        print(corrcoeffs_mean2)

    if ex34:
        """
        Exercise 3.4
        """
        # We move from 2 to 3 Gaussian components
        randomisations = 8
        n_iterations = 100
        loglikelihoods3 = np.zeros((n_iterations, randomisations))
        criterions3 = np.zeros((n_iterations, randomisations))
        labels3 = np.zeros((X.shape[ 0 ], randomisations))
        K = 3
        for j in np.arange(randomisations):
            np.random.seed(j)

            # initialisation
            init_means = np.repeat(np.mean(X, axis=0), K, axis=0).reshape(K, D)
            init_means += np.random.random_sample((K, D)) * 2.0 - 1.0
            init_weights = np.ones(K, dtype=float) / K
            init_covars = np.zeros((K, D, D))
            for k in np.arange(K):
                # the initialisation cannot lead to singularity.
                Sigma_k = np.random.random_sample(D) * 4.0 + 2.0
                init_covars[ k, :, : ] = np.diag(Sigma_k)
            init_covars = init_covars[ :, :, : ]

            # Fit a Gaussian mixture with EM using five components
            convergence_print = False
            for i in np.arange(0, n_iterations):
                # the data get quite big so we overwrite it every time
                gmm3 = mixture_models.MixtureModel(
                        n_components=K, n_iter=i,
                        means_init=init_means, weights_init=init_weights,
                        covars_init=init_covars,
                        random_state=np.random.seed(randomisations)
                )
                gmm3 = gmm3.fit(X)
                if gmm3.converged_ and not convergence_print:
                    print('converged at step {0}'.format(i))
                    convergence_print = True
                loglikelihoods3[ i, j ] = np.sum(gmm3.score_samples(X)[ 0 ])
                criterions3[ i, j ] = gmm3.bic(X)
            if not gmm3.converged_:
                print('no convergence in trial {0}'.format(j))
            # The final data labels:
            labels3[ :, j ] = gmm3.score_samples(X)[ 1 ].argmax(axis=1)

        # correlation coefficients
        corrcoeffs3 = np.zeros((K, randomisations))
        for l in np.arange(randomisations):
            for k in np.arange(K):
                submat = X[ labels3[ :, l ] == k, : ]
                corrcoeffs3[ k, l ] = \
                    np.corrcoef(submat[ :, 0 ], submat[ :, 1 ])[ 0, 1 ]
        corrcoeffs_mean3 = np.nanmean(corrcoeffs3, axis=1)
        print(corrcoeffs_mean3)

        # plotting
        # scatter plots
        fig342 = plt.figure()
        ax0 = fig342.add_subplot(221)
        idx = 3
        set0 = X[ labels3[ :, idx ] == 0, : ]
        set1 = X[ labels3[ :, idx ] == 1, : ]
        set2 = X[ labels3[ :, idx ] == 2, : ]
        ax0.scatter(set0[ :, 0 ], set0[ :, 1 ], color='b', s=6, alpha=0.75)
        ax0.scatter(set1[ :, 0 ], set1[ :, 1 ], color='r', s=6, alpha=0.75)
        ax0.scatter(set2[ :, 0 ], set2[ :, 1 ], color='g', s=6, alpha=0.75)
        ax0.set_xlabel(r'$x_1$-dimension')
        ax0.set_ylabel(r'$x_2$-dimension')
        ax1 = fig342.add_subplot(222)
        idx = 4
        set0 = X[ labels3[ :, idx ] == 0, : ]
        set1 = X[ labels3[ :, idx ] == 1, : ]
        set2 = X[ labels3[ :, idx ] == 2, : ]
        ax1.scatter(set0[ :, 0 ], set0[ :, 1 ], color='b', s=6, alpha=0.75)
        ax1.scatter(set1[ :, 0 ], set1[ :, 1 ], color='r', s=6, alpha=0.75)
        ax1.scatter(set2[ :, 0 ], set2[ :, 1 ], color='g', s=6, alpha=0.75)
        ax1.set_xlabel(r'$x_1$-dimension')
        ax1.set_ylabel(r'$x_2$-dimension')
        ax2 = fig342.add_subplot(223)
        idx = 1
        set0 = X[ labels3[ :, idx ] == 0, : ]
        set1 = X[ labels3[ :, idx ] == 1, : ]
        set2 = X[ labels3[ :, idx ] == 2, : ]
        ax2.scatter(set0[ :, 0 ], set0[ :, 1 ], color='b', s=6, alpha=0.75)
        ax2.scatter(set1[ :, 0 ], set1[ :, 1 ], color='r', s=6, alpha=0.75)
        ax2.scatter(set2[ :, 0 ], set2[ :, 1 ], color='g', s=6, alpha=0.75)
        ax2.set_xlabel(r'$x_1$-dimension')
        ax2.set_ylabel(r'$x_2$-dimension')
        ax3 = fig342.add_subplot(224)
        idc = 0
        set0 = X[ labels3[ :, idx ] == 0, : ]
        set1 = X[ labels3[ :, idx ] == 1, : ]
        set2 = X[ labels3[ :, idx ] == 2, : ]
        ax3.scatter(set0[ :, 0 ], set0[ :, 1 ], color='b', s=6, alpha=0.75)
        ax3.scatter(set1[ :, 0 ], set1[ :, 1 ], color='r', s=6, alpha=0.75)
        ax3.scatter(set2[ :, 0 ], set2[ :, 1 ], color='g', s=6, alpha=0.75)
        ax3.set_xlabel(r'$x_1$-dimension')
        ax3.set_ylabel(r'$x_2$-dimension')

        fig341 = plt.figure()
        ax1 = fig341.add_subplot(121)
        ax1.plot(np.arange(1, n_iterations), loglikelihoods3[ 1:, 1 ], 'r',
                 np.arange(1, n_iterations), loglikelihoods3[ 1:, 2 ], 'y--',
                 np.arange(1, n_iterations), loglikelihoods3[ 1:, 3 ], 'b',
                 np.arange(1, n_iterations), loglikelihoods3[ 1:, 4 ], 'y',
                 np.arange(1, n_iterations), loglikelihoods3[ 1:, 5 ], 'r--',
                 np.arange(1, n_iterations), loglikelihoods3[ 1:, 6 ], 'g--',
                 np.arange(1, n_iterations), loglikelihoods3[ 1:, 7 ], 'b--',
                 np.arange(1, n_iterations), loglikelihoods3[ 1:, 0 ], 'g'
                 )
        ax1.set_xlabel('Update step of EM-algorithm')
        ax1.set_ylabel('Log-likelihood of sample distribution')
        plt.title(r'$\mathrm{Loglikelihood}$')
        ax2 = fig341.add_subplot(122)
        ax2.plot(np.arange(1, n_iterations), criterions3[ 1:, 1 ], 'r',
                 np.arange(1, n_iterations), criterions3[ 1:, 2 ], 'y--',
                 np.arange(1, n_iterations), criterions3[ 1:, 3 ], 'b',
                 np.arange(1, n_iterations), criterions3[ 1:, 4 ], 'y',
                 np.arange(1, n_iterations), criterions3[ 1:, 5 ], 'r--',
                 np.arange(1, n_iterations), criterions3[ 1:, 6 ], 'g--',
                 np.arange(1, n_iterations), criterions3[ 1:, 7 ], 'b--',
                 np.arange(1, n_iterations), criterions3[ 1:, 0 ], 'g'
                 )
        ax2.set_xlabel('Update step of EM-algorithm')
        ax2.set_ylabel('Bayes information criterion')
        plt.title(r'$\mathrm{BIC}$')

        # We move from 3 to 4 Gaussian components
        randomisations = 8
        n_iterations = 100
        loglikelihoods4 = np.zeros((n_iterations, randomisations))
        criterions4 = np.zeros((n_iterations, randomisations))
        labels4 = np.zeros((X.shape[ 0 ], randomisations))
        K = 4
        for j in np.arange(randomisations):
            np.random.seed(j)

            # initialisation
            init_means = np.repeat(np.mean(X, axis=0), K, axis=0).reshape(K, D)
            init_means += np.random.random_sample((K, D)) * 2.0 - 1.0
            init_weights = np.ones(K, dtype=float) / K
            init_covars = np.zeros((K, D, D))
            for k in np.arange(K):
                # the initialisation cannot lead to singularity.
                Sigma_k = np.random.random_sample(D) * 4.0 + 2.0
                init_covars[ k, :, : ] = np.diag(Sigma_k)
            init_covars = init_covars[ :, :, : ]

            # Fit a Gaussian mixture with EM using five components
            convergence_print = False
            for i in np.arange(0, n_iterations):
                # the data get quite big so we overwrite it every time
                gmm4 = mixture_models.MixtureModel(
                        n_components=K, n_iter=i,
                        means_init=init_means, weights_init=init_weights,
                        covars_init=init_covars,
                        random_state=np.random.seed(randomisations)
                )
                gmm4 = gmm4.fit(X)
                if gmm4.converged_ and not convergence_print:
                    print('converged at step {0}'.format(i))
                    convergence_print = True
                loglikelihoods4[ i, j ] = np.sum(gmm4.score_samples(X)[ 0 ])
                criterions4[ i, j ] = gmm4.bic(X)
            if not gmm4.converged_:
                print('no convergence in trial {0}'.format(j))
            # The final data labels:
            labels4[ :, j ] = gmm4.score_samples(X)[ 1 ].argmax(axis=1)

        # correlation coefficients
        corrcoeffs4 = np.zeros((K, randomisations))
        for l in np.arange(randomisations):
            for k in np.arange(K):
                submat = X[ labels4[ :, l ] == k, : ]
                corrcoeffs4[ k, l ] = \
                    np.corrcoef(submat[ :, 0 ], submat[ :, 1 ])[ 0, 1 ]
        corrcoeffs_mean4 = np.nanmean(corrcoeffs4, axis=1)
        print(corrcoeffs_mean4)

        # plotting
        # scatter plots
        fig342 = plt.figure()
        ax0 = fig342.add_subplot(221)
        idx = 3
        set0 = X[ labels4[ :, idx ] == 0, : ]
        set1 = X[ labels4[ :, idx ] == 1, : ]
        set2 = X[ labels4[ :, idx ] == 2, : ]
        set3 = X[ labels4[ :, idx ] == 3, : ]
        ax0.scatter(set0[ :, 0 ], set0[ :, 1 ], color='b', s=6, alpha=0.75)
        ax0.scatter(set1[ :, 0 ], set1[ :, 1 ], color='r', s=6, alpha=0.75)
        ax0.scatter(set2[ :, 0 ], set2[ :, 1 ], color='g', s=6, alpha=0.75)
        ax0.scatter(set3[ :, 0 ], set3[ :, 1 ], color='y', s=6, alpha=0.75)
        ax0.set_xlabel(r'$x_1$-dimension')
        ax0.set_ylabel(r'$x_2$-dimension')
        ax1 = fig342.add_subplot(222)
        idx = 2
        set0 = X[ labels4[ :, idx ] == 0, : ]
        set1 = X[ labels4[ :, idx ] == 1, : ]
        set2 = X[ labels4[ :, idx ] == 2, : ]
        set3 = X[ labels4[ :, idx ] == 3, : ]
        ax1.scatter(set0[ :, 0 ], set0[ :, 1 ], color='b', s=6, alpha=0.75)
        ax1.scatter(set1[ :, 0 ], set1[ :, 1 ], color='r', s=6, alpha=0.75)
        ax1.scatter(set2[ :, 0 ], set2[ :, 1 ], color='g', s=6, alpha=0.75)
        ax1.scatter(set3[ :, 0 ], set3[ :, 1 ], color='y', s=6, alpha=0.75)
        ax1.set_xlabel(r'$x_1$-dimension')
        ax1.set_ylabel(r'$x_2$-dimension')
        ax2 = fig342.add_subplot(223)
        idx = 6
        set0 = X[ labels4[ :, idx ] == 0, : ]
        set1 = X[ labels4[ :, idx ] == 1, : ]
        set2 = X[ labels4[ :, idx ] == 2, : ]
        set3 = X[ labels4[ :, idx ] == 3, : ]
        ax2.scatter(set0[ :, 0 ], set0[ :, 1 ], color='b', s=6, alpha=0.75)
        ax2.scatter(set1[ :, 0 ], set1[ :, 1 ], color='r', s=6, alpha=0.75)
        ax2.scatter(set2[ :, 0 ], set2[ :, 1 ], color='g', s=6, alpha=0.75)
        ax2.scatter(set3[ :, 0 ], set3[ :, 1 ], color='y', s=6, alpha=0.75)
        ax2.set_xlabel(r'$x_1$-dimension')
        ax2.set_ylabel(r'$x_2$-dimension')
        ax3 = fig342.add_subplot(224)
        idx = 4
        set0 = X[ labels4[ :, idx ] == 0, : ]
        set1 = X[ labels4[ :, idx ] == 1, : ]
        set2 = X[ labels4[ :, idx ] == 2, : ]
        set3 = X[ labels4[ :, idx ] == 3, : ]
        ax3.scatter(set0[ :, 0 ], set0[ :, 1 ], color='b', s=6, alpha=0.75)
        ax3.scatter(set1[ :, 0 ], set1[ :, 1 ], color='r', s=6, alpha=0.75)
        ax3.scatter(set2[ :, 0 ], set2[ :, 1 ], color='g', s=6, alpha=0.75)
        ax3.scatter(set3[ :, 0 ], set3[ :, 1 ], color='y', s=6, alpha=0.75)
        ax3.set_xlabel(r'$x_1$-dimension')
        ax3.set_ylabel(r'$x_2$-dimension')

        fig341 = plt.figure()
        ax1 = fig341.add_subplot(121)
        ax1.plot(np.arange(1, n_iterations), loglikelihoods4[ 1:, 1 ], 'g--',
                 np.arange(1, n_iterations), loglikelihoods4[ 1:, 2 ], 'g',
                 np.arange(1, n_iterations), loglikelihoods4[ 1:, 3 ], 'b',
                 np.arange(1, n_iterations), loglikelihoods4[ 1:, 4 ], 'y',
                 np.arange(1, n_iterations), loglikelihoods4[ 1:, 5 ], 'r--',
                 np.arange(1, n_iterations), loglikelihoods4[ 1:, 6 ], 'r',
                 np.arange(1, n_iterations), loglikelihoods4[ 1:, 7 ], 'b--',
                 np.arange(1, n_iterations), loglikelihoods4[ 1:, 0 ], 'y--'
                 )
        ax1.set_xlabel('Update step of EM-algorithm')
        ax1.set_ylabel('Log-likelihood of sample distribution')
        plt.title(r'$\mathrm{Loglikelihood}$')
        ax2 = fig341.add_subplot(122)
        ax2.plot(np.arange(1, n_iterations), criterions4[ 1:, 1 ], 'g--',
                 np.arange(1, n_iterations), criterions4[ 1:, 2 ], 'g',
                 np.arange(1, n_iterations), criterions4[ 1:, 3 ], 'b',
                 np.arange(1, n_iterations), criterions4[ 1:, 4 ], 'y',
                 np.arange(1, n_iterations), criterions4[ 1:, 5 ], 'r--',
                 np.arange(1, n_iterations), criterions4[ 1:, 6 ], 'r',
                 np.arange(1, n_iterations), criterions4[ 1:, 7 ], 'b--',
                 np.arange(1, n_iterations), criterions4[ 1:, 0 ], 'y--'
                 )
        ax2.set_xlabel('Update step of EM-algorithm')
        ax2.set_ylabel('Bayes information criterion')
        plt.title(r'$\mathrm{BIC}$')
        plt.show()

    if ex35:
        # initialisation
        random_state = np.random.seed(3)
        n_iterations = 100
        loglikelihoods3 = np.zeros(n_iterations)
        criterions3 = np.zeros(n_iterations)
        labels3 = np.zeros(X.shape[ 0 ])
        K = 3
        init_means = np.repeat(np.mean(X, axis=0), K, axis=0).reshape(K, D)
        init_means += np.random.random_sample((K, D)) * 2.0 - 1.0
        init_weights = np.ones(K, dtype=float) / K
        init_covars = np.zeros((K, D, D))
        for k in np.arange(K):
            # the initialisation cannot lead to singularity.
            Sigma_k = np.random.random_sample(D) * 4.0 + 2.0
            init_covars[ k, :, : ] = np.diag(Sigma_k)
        init_covars = init_covars[ :, :, : ]
        
        # optimal solution
        convergence_print = False
        for i in np.arange(0, n_iterations):
            # the data get quite big so we overwrite it every time
            gmm3 = mixture_models.MixtureModel(
                    n_components=K, n_iter=i,
                    means_init=init_means, weights_init=init_weights,
                    covars_init=init_covars,
                    random_state=random_state
            )
            gmm3 = gmm3.fit(X)
            if gmm3.converged_ and not convergence_print:
                print('converged at step {0}'.format(i))
                convergence_print = True
            loglikelihoods3[ i ] = np.sum(gmm3.score_samples(X)[ 0 ])
            criterions3[ i ] = gmm3.bic(X)
        if not gmm3.converged_:
            print('no convergence in trial {0}'.format(j))
        # The final data labels:
        labels3[ : ] = gmm3.score_samples(X)[ 1 ].argmax(axis=1)

        """
        Exercise 3.5
        """
        # Here are our possible culprits:
        newsamples = np.array([ [ 11.85, 2.2, 0.5, 4.0 ],
                                [ 11.95, 3.1, 0.0, 1.0 ],
                                [ 12.00, 2.5, 0.0, 2.0 ],
                                [ 12.00, 3.0, 1.0, 6.3 ] ])
        alpha = gmm3.predict_proba(newsamples)
            # a priori assumption from two-cluster solution:
            # subject d has taken the substance
            # subject c has tampered with their values
