

from pylab import *
from numpy import *

close('all')
#  q1

# kernel = lambda  x1, x2, theta : \
# theta[0] * exp(- theta[1] / 2 * (x1 - x2)**2) + theta[2] + theta[3] * x1.T.dot(x2)
def kernel(x1, x2, theta):
    """
    Code for equation 3
    """
    sigma = np.zeros((len(x1), len(x2)))
    for idx, i in enumerate(x1):
        for jdx, j in enumerate(x2):
            sigma[ idx, jdx ] = \
                theta[ 0 ] * np.exp(- .5 * theta[ 1 ] * (i - j)**2) + \
                theta[ 2 ] + theta[ 3 ] * i * j
    return sigma
#q2
theta = ones(4)
N = 101
x = linspace(-1,1, N)
K = kernel(x, x, theta)
print(K.shape)
# q3
'''
K = N x N = 101 x 101
We can show semipositive definite by showing that all the eigenvalues are >= 0
'''
eigval, eigvec = linalg.eig(K)
# linalg returns complex values, but the complex part is zero
print(real(eigval))

# q4
from scipy.stats import multivariate_normal as mv
mu = zeros(len(x))
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
for idx, thetai in enumerate(thetas):
    K = kernel(x, x, thetai)
    prior = mv(mu, K, allow_singular=1)
    samples = prior.rvs(5)
    for i in samples:
        ax.flatten()[ idx ].plot(x, i)
    ax.flatten()[ idx ].set_title(thetai)
fig.tight_layout()
savefig('../Figures/ex1_test2.png')

# q6
xTrain = array([[-.5, .2, .3 , -.1]]).T
tTrain = array([[.5, -1, 3, -2.5]]).T
KTrain = kernel(xTrain, xTrain, theta)
beta   = 1

# q7
C      = KTrain + 1/beta * eye(len(xTrain))
xNew   = array([[0]])
c      = kernel(xNew, xNew, theta)
k      = kernel(xTrain, xNew, theta)
invC   = linalg.inv(C)
mu_new = k.T.dot(invC).dot(tTrain)
sigma_new = c - k.T.dot(invC).dot(k)

xTrainNew = vstack((xTrain, xNew))
KNew      = kernel(xTrainNew, xTrainNew, theta)

yNew      = mv(zeros(len(xTrainNew)), KNew)
samples   = yNew.rvs(50)

fig, ax = subplots()
ax.plot(xTrainNew, samples.T)
show()
print(samples)
'''
I believe we just computed the distribution of the input x = 0, i.e. we will expect that the target
will be around (mu_new) with variance (sigma_new), in this case it is around mu = .2.
Gaussian processes will just generate distributions over functions!
'''

#q8
'''
If we sample infinitely many points, the distance (x_n - x_m) will be nearly zero
(yielding the exponent to be one), hence the mean will note converge to zero, the covariance will converge
to identity. Since m(x_n+1) = k^TC^{-1} t, we will yield exactly the target vector
Adjusting the theta to be zero in front of the exponent will trivially
yield zero mean
'''
