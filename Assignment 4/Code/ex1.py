import numpy as np
from scipy.stats import multivariate_normal as mv
from pylab import *
from numpy.linalg import matrix_rank as rank
from numpy.linalg import cholesky as chol


# q1
def kernel_function(x, theta):
    """
    Code for equation 3
    """
    sigma = np.zeros((len(x), len(x)))
    for idx, i in enumerate(x):
        for jdx, j in enumerate(x):
            sigma[ idx, jdx ] = \
                theta[ 0 ] * np.exp(- .5 * theta[ 1 ] * (i - j)**2) + \
                theta[ 2 ] + theta[ 3 ] * i * j
    return sigma


# q2
# number of points to sample
N = 101
x = np.linspace(-1, 1, N)
theta = [ 1., 1., 1., 1. ]
K = kernel_function_alt(x, theta)

# q3
# dim:
# obvious: K.shape = len(linspace) * len(linspace)


# positive semidefinite:
# we can do a Cholesky decomposition and investigate the rank of L
# (rk K = dim im K - dim ker K = dim im K - dim ker L^T.L and
# dim ker L^T.L = dim ker L = # linear dependencies in L's rows)
# It suffices to show, that there is a Cholesky decomposition of K,
# since then there is L, such that L^T.L = K, thus K is a Gramian,
# thus it is p.sd. by construction.  (Symmetry is trivial, >= 0,
# because every component of rows of L is real, thus it's square
# is >= 0).

'''
semipositive definite proof: The only restriction on the kernel function is that the covariance matrix given by
(6.62) must be positive definite. If λi is an eigenvalue ofK,
then the corresponding eigenvalue of C will be λi + β−1.
It is therefore sufficient that the kernel matrix k(xn,xm) be
positive semidefinite for any pair of points xn and xm, so that λi ? 0,
because any eigenvalue λi that is zero will still give rise to a positive eigenvalue for
C because β> 0. [Bisschop 308]
'''

L = chol(K)
# this gives an error. is something wrong with the kernel?
l = rank(L)
# it would be surprising, if this was >= 0

# q4
mu = np.zeros(len(x))
# for some reason it errors on singularity?
prior = mv(mu, K, allow_singular=1)
samples = prior.rvs(5)

fig, ax = subplots(1, 1)
for i in samples:
    ax.plot(x, i)
savefig('../Figures/ex_1_test')

# q5
thetas = np.array([ [ 1, 4, 0, 0 ], [ 9, 4, 0, 0 ], [ 1, 64, 0, 0 ],
                    [ 1, 0.25, 0, 0 ], [ 1, 4, 10, 0 ], [ 1, 4, 0, 5 ] ])

# i want at most 3 rows
nRows = 3
# get the the number of columns
nCols = thetas.shape[ 0 ] // nRows

fig, ax = subplots(nrows=nRows, ncols=nCols, sharex='all')
for idx, theta in enumerate(thetas):
    K = kernel_function(x, theta)
    prior = mv(mu, K, allow_singular=1)
    samples = prior.rvs(5)
    for i in samples:
        ax.flatten()[ idx ].plot(x, i)
    ax.flatten()[ idx ].set_title(theta)
fig.tight_layout()
savefig('../Figures/ex1_test2.png')
show(block=1)

# q6
(x1 = −0.5, t1 = 0.5), (x2 = 0.2, t2 = −1), (x3 = 0.3, t3 = 3), (x4 = −0.1, t4 = −2.5)
