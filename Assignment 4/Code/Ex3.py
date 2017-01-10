"""
This is a default script that should be adapted to the respective purpose.
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
plt.clf()
#########################################################################
# some helper routines for Gaussian Mixture Models
#########################################################################


# def _covar_mstep_diag(gmm, X, responsibilities, weighted_X_sum, norm,
# 					  min_covar):
# 	"""Performing the covariance M step for diagonal cases"""
# 	avg_X2 = np.dot(responsibilities.T, X * X) * norm
# 	avg_means2 = gmm.means_**2
# 	avg_X_means = gmm.means_ * weighted_X_sum * norm
# 	return avg_X2 - 2 * avg_X_means + avg_means2 + min_covar
#
#
# def check_random_state(seed):
# 	"""Turn seed into a np.random.RandomState instance
#
# 	If seed is None, return the RandomState singleton used by np.random.
# 	If seed is an int, return a new RandomState instance seeded with seed.
# 	If seed is already a RandomState instance, return it.
# 	Otherwise raise ValueError.
# 	"""
# 	if seed is None or seed is np.random:
# 		return np.random.mtrand._rand
# 	if isinstance(seed, (numbers.Integral, np.integer)):
# 		return np.random.RandomState(seed)
# 	if isinstance(seed, np.random.RandomState):
# 		return seed
# 	raise ValueError('%r cannot be used to seed a '
# 					 'numpy.random.RandomState'
# 					 ' instance' % seed)
#
#
# def sample_gaussian(mean, covar, n_samples=1, random_state=None):
# 	"""Generate random samples from a Gaussian distribution.
# 	----------
# 	mean : array_like, shape (n_features,)
# 		Mean of the distribution.
# 	covar : array_like, (n_features)
# 	n_samples : int, optional
# 		Number of samples to generate. Defaults to 1.
# 	-------
# 	X : array, shape (n_features, n_samples)
# 		Randomly generated sample
# 	"""
# 	rng = check_random_state(random_state)
# 	n_dim = len(mean)
# 	rand = rng.randn(n_dim, n_samples)
# 	if n_samples == 1:
# 		rand.shape = (n_dim,)
# 	rand = np.dot(np.diag(np.sqrt(covar)), rand)
# 	return (rand.T + mean).T
#
#
# class GMM:
# 	"""Gaussian Mixture Model
# 	Representation of a Gaussian mixture model probability distribution.
# 	This class allows for easy evaluation of, sampling from, and
# 	maximum-likelihood estimation of the parameters of a GMM distribution.
# 	Initializes parameters such that every mixture component has zero
# 	mean and identity covariance.
# 	----------
# 	n_components : int, optional
# 		Number of mixture components. Defaults to 1.
# 	random_state: RandomState or an int seed (None by default)
# 		A random number generator instance
# 	min_covar : float, optional
# 		Floor on the diagonal of the covariance matrix to prevent
# 		overfitting.  Defaults to 1e-3.
# 	tol : float, optional
# 		Convergence threshold. EM iterations will stop when average
# 		gain in log-likelihood is below this threshold.  Defaults to 1e-3.
# 	n_iter : int, optional
# 		Number of EM iterations to perform.
# 	n_init : int, optional
# 		Number of initializations to perform. the best results is kept
# 	----------
# 	weights_ : array, shape (`n_components`,)
# 		This attribute stores the mixing weights for each mixture
# 		component.
# 	means_ : array, shape (`n_components`, `n_features`)
# 		Mean parameters for each mixture component.
# 	covars_ : array
# 		Covariance parameters for each mixture component.  The shape
# 		(n_components, n_features)
# 	converged_ : bool
# 		True when convergence was reached in fit(), False otherwise.
# 	--------
# 	Examples
# 	--------
# 	>>> import numpy as np
# 	>>> from sklearn import mixture
# 	>>> np.random.seed(1)
# 	>>> g = mixture.GMM(n_components=2)
# 	>>> # Generate random observations with two modes centered on 0
# 	>>> # and 10 to use for training.
# 	>>> obs = np.concatenate((np.random.randn(100, 1),
# 	...                       10 + np.random.randn(300, 1)))
# 	>>> g.fit(obs) # doctest: +NORMALIZE_WHITESPACE
# 	GMM(covariance_type='diag', init_params='wmc', min_covar=0.001,
# 			n_components=2, n_init=1, n_iter=100, params='wmc',
# 			random_state=None, tol=0.001)
# 	>>> np.round(g.weights_, 2)
# 	array([ 0.75,  0.25])
# 	>>> np.round(g.means_, 2)
# 	array([[ 10.05],
# 		   [  0.06]])
# 	>>> np.round(g.covars_, 2) #doctest: +SKIP
# 	array([[[ 1.02]],
# 		   [[ 0.96]]])
# 	>>> g.predict([[0], [2], [9], [10]]) #doctest: +ELLIPSIS
# 	array([1, 1, 0, 0]...)
# 	>>> np.round(g.score([[0], [2], [9], [10]]), 2)
# 	array([-2.19, -4.58, -1.75, -1.21])
# 	>>> # Refit the model on new data (initial parameters remain the
# 	>>> # same), this time with an even split between the two modes.
# 	>>> g.fit(20 * [[0]] +  20 * [[10]]) # doctest: +NORMALIZE_WHITESPACE
# 	GMM(covariance_type='diag', init_params='wmc', min_covar=0.001,
# 			n_components=2, n_init=1, n_iter=100, params='wmc',
# 			random_state=None, tol=0.001)
# 	>>> np.round(g.weights_, 2)
# 	array([ 0.5,  0.5])
# 	"""
#
# 	def __init__(self, n_components=1, random_state=None, tol=1e-3,
# 				 min_covar=1e-3, n_iter=100, n_init=1):
# 		self.n_components = n_components
# 		self.tol = tol
# 		self.min_covar = min_covar
# 		self.random_state = random_state
# 		self.n_iter = n_iter
# 		self.n_init = n_init
# 		self.weights_ = np.ones(self.n_components) / self.n_components
# 		self.converged_ = False
# 		if n_init < 1:
# 			raise ValueError('GMM estimation requires at least one run')
#
# 	def score_samples(self, X):
# 		"""Return the per-sample likelihood of the data under the model.
# 		Compute the log probability of X under the model and
# 		return the posterior distribution (responsibilities) of each
# 		mixture component for each element of X.
# 		----------
# 		X: array_like, shape (n_samples, n_features)
# 			List of n_features-dimensional data points. Each row
# 			corresponds to a single data point.
# 		-------
# 		logprob : array_like, shape (n_samples,)
# 			Log probabilities of each data point in X.
# 		responsibilities : array_like, shape (n_samples, n_components)
# 			Posterior probabilities of each mixture component for each
# 			observation
# 		"""
# 		X = np.asarray(X)
# 		if X.ndim == 1:
# 			X = X[ :, np.newaxis ]
# 		if X.size == 0:
# 			return np.array([ ]), np.empty((0, self.n_components))
# 		if X.shape[ 1 ] != self.means_.shape[ 1 ]:
# 			raise ValueError('The shape of X  is not compatible with '
# 							 'self')
#
# 		# compute the log-density
# 		lpr = (-0.5 * (
# 			n_dim * np.log(2 * np.pi) + np.sum(np.log(self.covars), 1) +
# 			np.sum((self.self.means**2) / self.covars, 1) - 2 * np.dot(X, (
# 				self.means / self.covars).T) + np.dot(X**2, (
# 					1.0 / self.covars).T)) + np.log(self.weights_))
#
# 		logprob = np.log(np.sum(np.exp(lpr), axis=1))
# 		responsibilities = np.exp(lpr - logprob[ :, np.newaxis ])
# 		return logprob, responsibilities
#
# 	def predict(self, X):
# 		"""Predict label for data.
# 		----------
# 		X : array-like, shape = [n_samples, n_features]
# 		-------
# 		C : array, shape = (n_samples,) component memberships
# 		"""
# 		logprob, responsibilities = self.score_samples(np.asarray(X))
# 		return responsibilities.argmax(axis=1)
#
# 	def sample(self, n_samples=1, random_state=None):
# 		"""Generate random samples from the model.
# 		----------
# 		n_samples : int, optional
# 			Number of samples to generate. Defaults to 1.
# 		-------
# 		X : array_like, shape (n_samples, n_features)
# 			List of samples
# 		"""
#
# 		if random_state is None:
# 			random_state = self.random_state
# 		random_state = check_random_state(random_state)
# 		weight_cdf = np.cumsum(self.weights_)
#
# 		X = np.empty((n_samples, self.means_.shape[ 1 ]))
# 		rand = random_state.rand(n_samples)
# 		# decide which component to use for each sample
# 		comps = weight_cdf.searchsorted(rand)
# 		# for each component, generate all needed samples
# 		for comp in range(self.n_components):
# 			# occurrences of current component in X
# 			comp_in_X = (comp == comps)
# 			# number of those occurrences
# 			num_comp_in_X = comp_in_X.sum()
# 			if num_comp_in_X > 0:
# 				cv = self.covars_[ comp ]
# 				X[ comp_in_X ] = sample_gaussian(
# 						self.means_[ comp ], cv,
# 						num_comp_in_X, random_state=random_state).T
# 		return X
#
# 	def fit_predict(self, X, y=None):
# 		"""Fit and then predict labels for data.
# 		Warning: due to the final maximization step in the EM algorithm,
# 		with low iterations the prediction may not be 100% accurate
# 		----------
# 		X : array-like, shape = [n_samples, n_features]
# 		-------
# 		C : array, shape = (n_samples,) component memberships
# 		"""
# 		return self._fit(X, y).argmax(axis=1)
#
# 	def _fit(self, X, y=None, do_prediction=False):
# 		"""Estimate model parameters with the EM algorithm.
# 		A initialization step is performed before entering the
# 		expectation-maximization (EM) algorithm. If you want to avoid
# 		this step, set the keyword argument init_params to the empty
# 		string '' when creating the GMM object. Likewise, if you would
# 		like just to do an initialization, set n_iter=0.
# 		----------
# 		X : array_like, shape (n, n_features)
# 			List of n_features-dimensional data points.  Each row
# 			corresponds to a single data point.
# 		-------
# 		responsibilities : array, shape (n_samples, n_components)
# 			Posterior probabilities of each mixture component for each
# 			observation.
# 		"""
#
# 		max_log_prob = -np.infty
# 		for init in range(self.n_init):
# 			# update the means
# 			self.means_ = cluster.KMeans(
# 					n_clusters=self.n_components,
# 					random_state=self.random_state).fit(
# 				X).cluster_centers_
# 			# update the weights/mixing coefficients
# 			self.weights_ = np.tile(1.0 / self.n_components,
# 									self.n_components)
# 			# update the covariants
# 			cv = np.cov(X.T) + self.min_covar * np.eye(X.shape[ 1 ])
# 			if not cv.shape:
# 				cv.shape = (1, 1)
# 			self.covars_ = np.tile(np.diag(cv), (self.n_components, 1))
#
# 			# EM algorithms
# 			current_log_likelihood = None
# 			self.converged_ = False
#
# 			for i in range(self.n_iter):
# 				prev_log_likelihood = current_log_likelihood
#
# 				# Expectation step
# 				log_likelihoods, responsibilities = self.score_samples(X)
# 				current_log_likelihood = log_likelihoods.mean()
#
# 				# Check for convergence.
# 				if prev_log_likelihood is not None:
# 					change = abs(
# 						current_log_likelihood - prev_log_likelihood)
# 					if change < self.tol:
# 						self.converged_ = True
# 						break
#
# 				# Maximization step
# 				self._do_mstep(X, responsibilities, self.min_covar)
#
# 			# if the results are better, keep it
# 			if self.n_iter:
# 				if current_log_likelihood > max_log_prob:
# 					max_log_prob = current_log_likelihood
# 					best_params = {'weights': self.weights_,
# 								   'means':   self.means_,
# 								   'covars':  self.covars_}
#
# 		if self.n_iter:
# 			self.covars_ = best_params[ 'covars' ]
# 			self.means_ = best_params[ 'means' ]
# 			self.weights_ = best_params[ 'weights' ]
# 		return responsibilities
#
# 	def fit(self, X, y=None):
# 		"""Estimate model parameters with the EM algorithm.
# 		A initialization step is performed before entering the
# 		expectation-maximization (EM) algorithm. If you want to avoid
# 		this step, set the keyword argument init_params to the empty
# 		string '' when creating the GMM object. Likewise, if you would
# 		like just to do an initialization, set n_iter=0.
# 		----------
# 		X : array_like, shape (n, n_features)
# 			List of n_features-dimensional data points.  Each row
# 			corresponds to a single data point.
# 		-------
# 		self
# 		"""
# 		self._fit(X, y)
# 		return self
#
# 	def _do_mstep(self, X, responsibilities, min_covar=0):
# 		""" Perform the Mstep of the EM algorithm and return the class
# 		weights
# 		"""
# 		weights = responsibilities.sum(axis=0)
# 		weighted_X_sum = np.dot(responsibilities.T, X)
# 		inverse_weights = 1.0 / (weights[ :, np.newaxis ] + 10 * EPS)
#
# 		# re-estimate mixing coefficients
# 		self.weights_ = (weights / (weights.sum() + 10 * EPS) + EPS)
# 		# re-estimate means
# 		self.means_ = weighted_X_sum * inverse_weights
# 		# re-estimate covariance
# 		self.covars_ = _covar_mstep_diag(
# 			self, X, responsibilities, weighted_X_sum,
# 			inverse_weights, min_covar)
# 		return weights
#
# 	def _n_parameters(self):
# 		"""Return the number of free parameters in the model."""
# 		ndim = self.means_.shape[ 1 ]
# 		cov_params = self.n_components * ndim
# 		mean_params = ndim * self.n_components
# 		return int(cov_params + mean_params + self.n_components - 1)
#
# 	def bic(self, X):
# 		"""Bayesian information criterion for the current model fit
# 		and the proposed data
# 		----------
# 		X : array of shape(n_samples, n_dimensions)
# 		-------
# 		bic: float (the lower the better)
# 		"""
# 		logprob, _ = self.score_samples(X)
# 		return (-2 * logprob.sum() +
# 				self._n_parameters() * np.log(X.shape[ 0 ]))
#
# 	def aic(self, X):
# 		"""Akaike information criterion for the current model fit
# 		and the proposed data
# 		----------
# 		X : array of shape(n_samples, n_dimensions)
# 		-------
# 		aic: float (the lower the better)
# 		"""
# 		logprob, _ = self.score_samples(X)
# 		return - 2 * logprob.sum() + 2 * self._n_parameters()


if __name__ == '__main__':
    # let us first load the data:
    os.chdir('../Data/')
    X = np.loadtxt('a011_mixdata.txt')
    os.chdir('../Code/')
    N, D = X.shape
    # n = number of datapoints
    # D = number of features

    ex31 = False
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

    ex32 = False
    if ex32:
        """
        Exercise 3.2
        """
        EPS = np.finfo(float).eps

        """
        Gaussian Mixture Models.
        """
        # TODO: replace this with the commented out class above!!!!
        from sklearn import mixture

        # set up some stuff for the gaussian mixture models
        n_iterations = 100
        K = 2
        np.random.seed(1)
        # initialisation
        init_means = np.repeat(np.mean(X, axis=0), K, axis=0).reshape(K, D)
        init_means += np.random.random_sample((K, D)) * 2.0 - 1.0
        init_weights = np.ones(K, dtype=float) / K
        init_precisions = np.zeros((K, D, D))
        for k in np.arange(K):
            # the initialisation cannot lead to singularity.
            Sigma_k = np.random.random_sample(D) * 4.0 + 2.0
            init_precisions[ k, :, : ] = np.diag(1. / Sigma_k)
        init_precisions = init_precisions[ :, :, : ]

        # Fit a Gaussian mixture with EM using five components
        loglikelihoods = np.zeros(n_iterations)
        criterions = np.zeros(n_iterations)
        convergence_print = False
        for i in np.arange(0, n_iterations):
            gmm = mixture.GaussianMixture(n_components=K,
                                          covariance_type='full',
                                          means_init=init_means,
                                          weights_init=init_weights,
                                          precisions_init=init_precisions,
                                          random_state=np.random.seed(1),
                                          max_iter=1,
                                          tol=1e-8)
            gmm.max_iter = i + 1
            gmm = gmm.fit(X)
            if gmm.converged_ and not convergence_print:
                print('converged at step {0}'.format(i))
                convergence_print = True
            loglikelihoods[ i ] = gmm.score(X)
            criterions[ i ] = gmm.bic(X)

        # The final data labels:
        labels = gmm.predict(X)

        # TODO: We could make this a flash movie...

        # let's plot the change in the loglikelihood over iterations
        fig321 = plt.figure()
        ax1 = fig321.add_subplot(121)
        ax1.plot(np.arange(n_iterations), loglikelihoods)
        ax1.set_xlabel('Update step of EM-algorithm')
        ax1.set_ylabel('Log-likelihood of sample distribution')
        ax2 = fig321.add_subplot(122)
        ax2.plot(np.arange(n_iterations), criterions)
        ax2.set_xlabel('1st variable')
        ax2.set_ylabel('2th variable')

        # TODO: make this a function
        # Now plot the requested colour-coded first two variables
        fig322 = plt.figure()
        ax3 = fig322.add_subplot(111)
        set0 = X[ labels == 0, : ]
        set1 = X[ labels == 1, : ]
        ax3.scatter(set0[ :, 0 ], set0[ :, 1 ], color='b')
        ax3.scatter(set1[ :, 0 ], set1[ :, 1 ], color='r')
        ax3.set_xlabel('Update step of EM-algorithm')
        ax3.set_ylabel('Log-likelihood of sample distribution')
        plt.show()

    ex33 = False
    if ex33:
        """
        Exercise 3.3
        """
        # The above version was a test run. Now some different initialisations:
        randomisations = 8
        n_iterations = 50
        loglikelihoods = np.zeros(n_iterations, randomisations)
        criterions = np.zeros(n_iterations, randomisations)
        labels = np.zeros(X.shape[ 0 ], randomisations)
        K = 2
        for j in np.arange(randomisations):
            np.random.seed(j)

            # initialisation
            init_means = np.repeat(np.mean(X, axis=0), K, axis=0).reshape(K, D)
            init_means += np.random.random_sample((K, D)) * 2.0 - 1.0
            init_weights = np.ones(K, dtype=float) / K
            init_precisions = np.zeros((K, D, D))
            for k in np.arange(K):
                # the initialisation cannot lead to singularity.
                Sigma_k = np.random.random_sample(D) * 4.0 + 2.0
                init_precisions[ k, :, : ] = np.diag(1. / Sigma_k)
            init_precisions = init_precisions[ :, :, : ]

            # Fit a Gaussian mixture with EM using five components
            convergence_print = False
            for i in np.arange(0, n_iterations):
                # the data get quite big so we overwrite it every time
                gmm = mixture.GaussianMixture(
                        n_components=K, covariance_type='full', max_iter=1,
                        means_init=init_means, weights_init=init_weights,
                        precisions_init=init_precisions, tol=1e-8,
                        random_state=np.random.seed(randomisations)
                        )
                gmm.max_iter = i + 1
                gmm = gmm.fit(X)
                if gmm.converged_ and not convergence_print:
                    print('converged at step {0}'.format(i))
                    convergence_print = True
                loglikelihoods[ i, j ] = gmm.score(X)
                criterions[ i, j ] = gmm.bic(X)

            # The final data labels:
            labels[ :, j ] = gmm.predict(X)

            # TODO: correlations

    ex34 = True
    if ex34:
        """
        Exercise 3.4
        """
        # We move from 2 to 3 Gaussian components
        randomisations = 4
        n_iterations = 50
        loglikelihoods = np.zeros(n_iterations, randomisations)
        criterions = np.zeros(n_iterations, randomisations)
        labels = np.zeros(X.shape[ 0 ], randomisations)
        K = 3
        for j in np.arange(randomisations):
            np.random.seed(j)

            # initialisation
            init_means = np.repeat(np.mean(X, axis=0), K, axis=0).reshape(K, D)
            init_means += np.random.random_sample((K, D)) * 2.0 - 1.0
            init_weights = np.ones(K, dtype=float) / K
            init_precisions = np.zeros((K, D, D))
            for k in np.arange(K):
                # the initialisation cannot lead to singularity.
                Sigma_k = np.random.random_sample(D) * 4.0 + 2.0
                init_precisions[ k, :, : ] = np.diag(1. / Sigma_k)
            init_precisions = init_precisions[ :, :, : ]

            # Fit a Gaussian mixture with EM using five components
            convergence_print = False
            for i in np.arange(0, n_iterations):
                # the data get quite big so we overwrite it every time
                gmm3 = mixture.GaussianMixture(
                        n_components=K, covariance_type='full', max_iter=1,
                        means_init=init_means, weights_init=init_weights,
                        precisions_init=init_precisions, tol=1e-8,
                        random_state=np.random.seed(randomisations)
                )
                gmm3.max_iter = i + 1
                gmm3 = gmm3.fit(X)
                if gmm3.converged_ and not convergence_print:
                    print('converged at step {0}'.format(i))
                    convergence_print = True
                loglikelihoods[ i, j ] = gmm3.score(X)
                criterions[ i, j ] = gmm3.bic(X)

            # The final data labels:
            labels[ :, j ] = gmm3.predict(X)

            # TODO: correlations, plotting

        # We move from 3 to 4 Gaussian components
        randomisations = 4
        n_iterations = 50
        loglikelihoods = np.zeros(n_iterations, randomisations)
        criterions = np.zeros(n_iterations, randomisations)
        labels = np.zeros(X.shape[ 0 ], randomisations)
        K = 4
        for j in np.arange(randomisations):
            np.random.seed(j)

            # initialisation
            init_means = np.repeat(np.mean(X, axis=0), K, axis=0).reshape(K, D)
            init_means += np.random.random_sample((K, D)) * 2.0 - 1.0
            init_weights = np.ones(K, dtype=float) / K
            init_precisions = np.zeros((K, D, D))
            for k in np.arange(K):
                # the initialisation cannot lead to singularity.
                Sigma_k = np.random.random_sample(D) * 4.0 + 2.0
                init_precisions[ k, :, : ] = np.diag(1. / Sigma_k)
            init_precisions = init_precisions[ :, :, : ]

            # Fit a Gaussian mixture with EM using five components
            convergence_print = False
            for i in np.arange(0, n_iterations):
                # the data get quite big so we overwrite it every time
                gmm4 = mixture.GaussianMixture(
                        n_components=K, covariance_type='full', max_iter=1,
                        means_init=init_means, weights_init=init_weights,
                        precisions_init=init_precisions, tol=1e-8,
                        random_state=np.random.seed(randomisations)
                )
                gmm4.max_iter = i + 1
                gmm4 = gmm4.fit(X)
                if gmm4.converged_ and not convergence_print:
                    print('converged at step {0}'.format(i))
                    convergence_print = True
                loglikelihoods[ i, j ] = gmm4.score(X)
                criterions[ i, j ] = gmm4.bic(X)

            # The final data labels:
            labels[ :, j ] = gmm4.predict(X)

            # TODO: correlations, plotting

    ex35 = True
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