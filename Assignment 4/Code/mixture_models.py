"""
Gaussian/Bernoulli Mixture Models.
"""

import numpy as np
from scipy import linalg
from sklearn.externals.six.moves import zip
from itertools import product

EPS = np.finfo(float).eps

class MixtureModel:
    def __init__(self,
                 means_init,
                 weights_init,
                 covars_init=None,
                 n_components=1,
                 random_state=None,
                 errtol=1e-8,
                 min_covar=1e-8,
                 n_iter=1,
                 distrib='Gaussian'):
        self.distrib = distrib
        self.n_components = n_components
        self.errtol = errtol
        self.min_covar = min_covar
        self.random_state = random_state
        self.n_iter = n_iter
        self.is_fitted = False
        self.converged_ = False
        self.means_ = means_init
        self.weights_ = weights_init
        self.covars_ = covars_init

    def score_samples(self, X):
        """
        This includes computing the responsibilities, so the E-Step
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X[:, np.newaxis]
        if X.size == 0:
            return np.array([]), np.empty((0, self.n_components))
        if X.shape[1] != self.means_.shape[1]:
            raise ValueError('The shape of X is not compatible with self')

        if self.distrib == 'Gaussian':
            lpr = _log_multivariate_normal_density(
                    X, self.means_, self.covars_) + np.log(self.weights_)
        elif self.distrib == 'Bernoulli':
            lpr = _log_Bernoulli_density(
                    X, self.means_) + np.log(self.weights_)
        else:
            raise ValueError('Did not find a distribution to score samples on')
        logprob = np.log(np.sum(np.exp(lpr), axis=1))
        # (9.23) / (9.56)
        responsibilities = np.exp(lpr - logprob[:, np.newaxis])
        return logprob, responsibilities

    def predict_proba(self, X):
        logprob, responsibilities = self.score_samples(X)
        return responsibilities

    def _fit(self, X, y=None, do_prediction=False):
        # initialization step
        X = np.asarray(X, dtype=np.float64)
        if X.shape[0] < self.n_components:
            raise ValueError(
                'GMM estimation with %s components, but got only %s samples' %
                (self.n_components, X.shape[0]))

        max_log_prob = -np.infty

        # EM algorithms
        current_log_likelihood = None
        self.converged_ = False

        for i in np.arange(self.n_iter):
            prev_log_likelihood = current_log_likelihood
            # Expectation step
            log_likelihoods, responsibilities = self.score_samples(X)
            current_log_likelihood = log_likelihoods.mean()

            # Check for convergence.
            if prev_log_likelihood is not None:
                change = abs(current_log_likelihood - prev_log_likelihood)
                if change < self.errtol:
                    self.converged_ = True
                    break

            # Maximization step
            self._do_mstep(X, responsibilities, self.min_covar)

        # if the results are better, keep it
        if self.n_iter:
            if current_log_likelihood > max_log_prob:
                if self.distrib == 'Gaussian':
                    best_params = {'weights': self.weights_,
                                   'means': self.means_,
                                   'covars': self.covars_}
                elif self.distrib == 'Bernoulli':
                    best_params = {'weights': self.weights_,
                                   'means':   self.means_}
        if self.n_iter:
            if self.distrib == 'Gaussian':
                self.covars_ = best_params['covars']
            self.means_ = best_params['means']
            self.weights_ = best_params['weights']
        else:
            responsibilities = np.zeros((X.shape[0], self.n_components))
        self.is_fitted = True
        return responsibilities

    def fit(self, X, y=None):
        """ The public version to call the fit and get responsibilities"""
        self._fit(X, y)
        return self

    def _do_mstep(self, X, responsibilities, min_covar=0):
        """ Perform the Mstep of the EM algorithm and return the class weights
        """
        weights = responsibilities.sum(axis=0)
        weighted_X_sum = np.dot(responsibilities.T, X)
        inverse_weights = 1.0 / (weights[ :, np.newaxis ] + 10 * EPS)

        if self.distrib == 'Gaussian':
            # (9.26)
            self.weights_ = (weights / (weights.sum() + 10 * EPS) + EPS)
            # (9.24)
            self.means_ = weighted_X_sum * inverse_weights
            # (9.25)
            self.covars_ = _covar_mstep(self, X, responsibilities, weighted_X_sum,
                    inverse_weights, min_covar)
        elif self.distrib == 'Bernoulli':
            # (9.60)
            self.weights_ = weights / weights.sum()
            # (9.59)
            self.means_ = weighted_X_sum * inverse_weights
        return weights

    def _n_parameters(self):
        """Return the number of free parameters in the model."""
        ndim = self.means_.shape[1]
        cov_params = self.n_components * ndim * (ndim + 1) / 2.
        mean_params = ndim * self.n_components
        return int(cov_params + mean_params + self.n_components - 1)

    def bic(self, X):
        return (-2 * self.score_samples(X)[ 0 ].sum() +
                self._n_parameters() * np.log(X.shape[0]))

#########################################################################
# some helper routines
#########################################################################


def _log_multivariate_normal_density(X, means, covars, min_covar=1.e-7):
    """Log probability for covariance matrices."""
    n_samples, n_dim = X.shape
    nmix = len(means)
    log_prob = np.empty((n_samples, nmix))
    for c, (mu, cv) in enumerate(zip(means, covars)):
        try:
            cv_chol = linalg.cholesky(cv, lower=True)
        except linalg.LinAlgError:
            # The model is most probably stuck in a component with too
            # few observations, we need to reinitialize this components
            try:
                cv_chol = linalg.cholesky(cv + min_covar * np.eye(n_dim),
                                          lower=True)
            except linalg.LinAlgError:
                raise ValueError("'covars' must be symmetric, "
                                 "positive-definite")
        # find the precision via determinant formula and cholesky decomposition
        cv_log_det = 2 * np.sum(np.log(np.diagonal(cv_chol)))
        cv_sol = linalg.solve_triangular(cv_chol, (X - mu).T, lower=True).T
        log_prob[:, c] = - .5 * (np.sum(cv_sol ** 2, axis=1) +
                                 n_dim * np.log(2 * np.pi) + cv_log_det)
    return log_prob


def _log_Bernoulli_density(X, means):
    """Log probability for covariance matrices."""
    n_samples, n_dim = X.shape
    nmix = len(means)
    log_prob = np.empty((n_samples, nmix))
    for n, k in product(n_samples, np.arange(nmix)):
        # need log (p(x_n | mu_k)), that is, log( 9.48 )
        log_prob[ n, k ] = np.sum(np.log(means[ k ]) * X[ n, : ] + np.log(
            np.ones_like(means[ k ] - means[ k ])) * (
            np.ones_like(X[ n, : ]) - X[ n, : ]))
    return log_prob


def _covar_mstep(gmm, X, responsibilities, weighted_X_sum, norm, min_covar):
    """Performing the covariance M step"""
    n_features = X.shape[1]
    cv = np.empty((gmm.n_components, n_features, n_features))
    for c in np.arange(gmm.n_components):
        post = responsibilities[:, c]
        mu = gmm.means_[c]
        diff = X - mu
        with np.errstate(under='ignore'):
            # (9.25)
            avg_cv = np.dot(post * diff.T, diff) / (post.sum() + 10 * EPS)
        cv[c] = avg_cv + min_covar * np.eye(n_features)
    return cv