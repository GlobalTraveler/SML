import numpy as np
from scipy.stats import multivariate_normal as mv
from pylab import *


# q1
def kernel_function(x, theta):
    '''
    Code for equation 3
    '''

    sigma = np.zeros((len(x), len(x)))
    for idx, i in enumerate(x):
        for jdx, j in enumerate(x):
            sigma[idx, jdx] = \
            theta[0] * np.exp(- .5 * theta[1] * (i -  j)**2 ) + \
             theta[2] + theta[3] * i * j
    return sigma


# q2
# number of points to sample
N = 101
x = np.linspace(-1, 1, N)
theta = [1,1,1,1]
K = kernel_function(x, theta)

# q4
mu = np.zeros(len(x))
# for some reason it errors on singularity?
prior = mv(mu, K, allow_singular = 1)
samples = prior.rvs(5)

fig, ax = subplots(1,1)
for i in samples:
    ax.plot(x, i)
savefig('../Figures/ex_1_test')

thetas = np.array([\
[1, 4, 0, 0], \
[9, 4, 0, 0],\
[1, 64, 0, 0],\
[1, 0.25, 0,0],\
[1, 4, 10, 0],\
[1, 4, 0, 5] ])

# i want at most 3 rows
nRows = 3
# get the the number of columns
nCols = thetas.shape[0] // nRows

fig, ax = subplots(nrows = nRows, ncols = nCols, sharex = 'all')
for idx, theta in enumerate(thetas):
    K = kernel_function(x, theta)
    prior = mv(mu, K, allow_singular = 1)
    samples = prior.rvs(5)
    for i in samples:
        ax.flatten()[idx].plot(x, i)
    ax.flatten()[idx].set_title(theta)
fig.tight_layout()
savefig('../Figures/ex1_test2.png')
show(block = 1)
