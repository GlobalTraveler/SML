


class mlp(object):
    def __init__(self, X, t, eta = 1e-1, gamma = 0, M = 8, K = 1):
        # learning rate / momentum rate
        self.eta        = eta
        self.gamma      = gamma
        # Layer dimensions; input, hidden, output
        self.D          = X.shape[1] + 1
        self.M          = M
        self.K          = K
        # add bias node to input
        self.X          = hstack( (X, ones(( X.shape[0], 1 ) ) ) )
        self.targets    = t
        # weights; hidden and output
        wh = random.rand(self.D, M) - 1/2
        wo = random.rand(M, K) - 1/2
        self.layers = [wh, wo]
        # activation functions:
        self.func = lambda x: tanh(x)
        self.dfunc = lambda x:  1 - tanh(x) **2

    def forwardSingle(self, xi):
        ''' Performs a single forward pass in the network'''
        layerOutputs = [ []  for j in self.layers ]
        #forward pass
        a = xi.dot(self.layers[0]); z = np.tanh(a); y = z.dot(self.layers[1])
        layerOutputs[0].append(z);
        layerOutputs[1].append(y)
        return layerOutputs
    def backwardsSingle(self, ti, xi):
        '''Backprop + update of weights'''
        nLayers = len(self.layers) - 2
        forwardPass = self.forwardSingle(xi)

        # loop through all the predictions and update with gradients
        dk = ti - forwardPass[-1][0]
        squaredError = dk**2
        # compute hidden activation; note elementwise product!!
        dj = self.dfunc(forwardPass[0][0]) * (self.layers[-1].dot(dk.T).T)
        # update the weights
        E1 = forwardPass[0][0].T.dot(dk)
        E2 = xi.T.dot(dj)
        self.layers[-1] += self.eta * E1 + self.gamma * self.layers[-1]
        self.layers[0]  += self.eta * E2 + self.gamma * self.layers[0]
        return squaredError
    def train(self, num):
        num = int(num) # for scientific notation
        MSE = zeros(num) # mean squared error
        for iter in range(num):
            error = 0
            for idx, (ti, xi) in enumerate(zip(self.targets, self.X)):
                xi = array(xi, ndmin = 2)
                error += self.backwardsSingle(ti, xi)
            MSE[iter] = error / num
        return MSE


from pylab import *
from numpy import *
from scipy.stats import multivariate_normal as mv

tmp = arange(-2, 2, .1)
x, y = meshgrid(tmp, tmp)

mu = [0,0]
sigma = eye(len(mu)) * 2/5
dist = mv(mu, sigma)

X = vstack((x.flatten(), y.flatten())).T
Y = dist.pdf(X).reshape(x.shape)  * 3
targets = array(Y.flatten(), ndmin = 2).T

# shuffle the indices
idx = np.random.permutation(range(len(targets)))
# shuffle the data
shuffleX = X[idx,:]
# shuffle the targets as well (same indices)
shuffleTargets = targets[idx]

model = mlp(shuffleX, shuffleTargets, eta = 1e-2)
mse = model.train(1e2)
print(mse)
fig, ax = subplots(); ax.plot(mse)
show()
