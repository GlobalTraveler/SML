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
    ex32 = True
    ex33 = False
    ex34 = False
    ex35 = False

    if ex31:
        """
        Exercise 3.1
        """
        fig = plt.figure()
        ax = fig.add_subplot(211, projection='3d')
        ax.scatter(X[ :, 0 ], X[ :, 1 ], X[ :, 2 ], c='c', marker='o')
        ax.set_xlabel('1st variable')
        ax.set_xlabel('2nd variable')
        ax.set_xlabel('3rd variable')
        ax2 = fig.add_subplot(212, projection='3d')
        ax2.scatter(X[ :, 0 ], X[ :, 2 ], X[ :, 3 ], c='c', marker='o')
        ax2.set_xlabel('1st variable')
        ax2.set_xlabel('2th variable')
        ax2.set_xlabel('4th variable')
        plt.show()

    if ex32:
        """
        Exercise 3.2
        """
        # TODO: replace this with the commented out class above!!!!

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

        # TODO: We could make this a flash movie...

        # let's plot the change in the loglikelihood over iterations
        fig321 = plt.figure()
        ax1 = fig321.add_subplot(121)
        ax1.plot(np.arange(1, n_iterations), loglikelihoods[ 1: ])
        ax1.set_xlabel('Update step of EM-algorithm')
        ax1.set_ylabel('Log-likelihood of sample distribution')
        ax2 = fig321.add_subplot(122)
        ax2.plot(np.arange(1, n_iterations), criterions[ 1: ])
        ax2.set_xlabel('Update step of EM-algorithm')
        ax2.set_ylabel('Bayes information criterion')

        # TODO: make this a function, add labels
        # Now plot the requested colour-coded first two variables
        fig322 = plt.figure()
        ax3 = fig322.add_subplot(111)
        set0 = X[ labels == 0, : ]
        set1 = X[ labels == 1, : ]
        ax3.scatter(set0[ :, 0 ], set0[ :, 1 ], color='b')
        ax3.scatter(set1[ :, 0 ], set1[ :, 1 ], color='r')
        ax3.set_xlabel('x_1-dimension')
        ax3.set_ylabel('x_2-dimension')
        plt.show()

    if ex33:
        """
        Exercise 3.3
        """
        # The above version was a test run. Now some different initialisations:
        randomisations = 8
        n_iterations = 50
        loglikelihoods2 = np.zeros(n_iterations, randomisations)
        criterions2 = np.zeros(n_iterations, randomisations)
        labels2 = np.zeros(X.shape[ 0 ], randomisations)
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
                gmm = mixture.GaussianMixture(
                        n_components=K, covariance_type='full', max_iter=i,
                        means_init=init_means, weights_init=init_weights,
                        covars_init=init_covars, tol=1e-8,
                        random_state=np.random.seed(randomisations)
                        )
                gmm = gmm.fit(X)
                if gmm.converged_ and not convergence_print:
                    print('converged at step {0}'.format(i))
                    convergence_print = True
                loglikelihoods2[ i, j ] = gmm.score_samples(X)[ 0 ]
                criterions2[ i, j ] = gmm.bic(X)

            # The final data labels:
            labels2 = gmm.score_samples(X)[ 1 ].argmax(axis=1)

            # TODO: correlations

    if ex34:
        """
        Exercise 3.4
        """
        # We move from 2 to 3 Gaussian components
        randomisations = 4
        n_iterations = 50
        loglikelihoods3 = np.zeros(n_iterations, randomisations)
        criterions3 = np.zeros(n_iterations, randomisations)
        labels3 = np.zeros(X.shape[ 0 ], randomisations)
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
                gmm3 = mixture.GaussianMixture(
                        n_components=K, covariance_type='full', max_iter=i,
                        means_init=init_means, weights_init=init_weights,
                        covars_init=init_covars, tol=1e-8,
                        random_state=np.random.seed(randomisations)
                )
                gmm3 = gmm3.fit(X)
                if gmm3.converged_ and not convergence_print:
                    print('converged at step {0}'.format(i))
                    convergence_print = True
                loglikelihoods3[ i, j ] = gmm3.score_samples(X)[ 0 ]
                criterions3[ i, j ] = gmm3.bic(X)

            # The final data labels:
            labels3 = gmm.score_samples(X)[ 1 ].argmax(axis=1)

            # TODO: correlations, plotting

        # We move from 3 to 4 Gaussian components
        randomisations = 4
        n_iterations = 50
        loglikelihoods4 = np.zeros(n_iterations, randomisations)
        criterions4 = np.zeros(n_iterations, randomisations)
        labels4 = np.zeros(X.shape[ 0 ], randomisations)
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
                gmm4 = mixture.GaussianMixture(
                        n_components=K, covariance_type='full', max_iter=i,
                        means_init=init_means, weights_init=init_weights,
                        covars_init=init_covars, tol=1e-8,
                        random_state=np.random.seed(randomisations)
                )
                gmm4 = gmm4.fit(X)
                if gmm4.converged_ and not convergence_print:
                    print('converged at step {0}'.format(i))
                    convergence_print = True
                loglikelihoods4[ i, j ] = gmm4.score_samples(X)[ 0 ]
                criterions4[ i, j ] = gmm4.bic(X)

            # The final data labels:
            labels4 = gmm.score_samples(X)[ 1 ].argmax(axis=1)

            # TODO: correlations, plotting

    if ex35:
        """
        Exercise 3.5
        """
        # Here are our possible culprits:
        newsamples = np.array([[ 11.85, 2.2, 0.5, 4.0 ],
                               [ 11.95, 3.1, 0.0, 1.0 ],
                               [ 12.00, 2.5, 0.0, 2.0 ],
                               [ 12.00, 3.0, 1.0, 6.3 ]])
        if ex32:
            gmm.predict_proba(newsamples)
        if ex33:
            gmm3.predict_proba(newsamples)
        if ex34:
            gmm4.predict_proba(newsamples)
        # a priori assumption from two-cluster solution:
        # subject d has taken the substance
        # subject c has tampered with their values
