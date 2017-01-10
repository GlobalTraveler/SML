
# coding: utf-8

# # Exercise 2 - Neural network regression
# 

# # 1

# In[1]:

import seaborn as sb
from scipy.stats import multivariate_normal as mv
from mpl_toolkits.mplot3d import Axes3D
from pylab import *
from numpy import *
# q1
# create the target distribution
# sample at .1 interval
tmp = arange(-2, 2, .1)
x, y = meshgrid(tmp, tmp)

mu = [0,0]
sigma = eye(len(mu)) * 2/5
dist = mv(mu, sigma)

X = vstack((x.flatten(), y.flatten())).T
Y = dist.pdf(X).reshape(x.shape)  * 3
targets = array(Y.flatten(), ndmin = 2).T

fig, ax = subplots(1, 1, subplot_kw = {'projection': '3d'})
ax.plot_surface(x, y, targets.reshape(x.shape))
ax.set_xlabel('$x_1$', labelpad = 20)
ax.set_ylabel('$x_2$', labelpad = 20)
ax.set_zlabel('pdf', labelpad =20)
sb.set_context('poster')
sb.set_style('white')
show()


# # 2

# In[4]:

class mlp(object):
    '''
    Multi-layered-pereptron
    K = output nodes
    M = hidden nodes
    Assumes the input data X is samples x feature dimension
    Returns:
        prediction and error
    '''
    def __init__(self, X, t, eta = 1e-1, gamma = .0, M = 8, K = 1):
        # learning rate / momentum rate
        self.eta        = eta
        self.gamma      = gamma
        # Layer dimensions; input, hidden, output
        self.D          = D =  X.shape[1] + 1
        self.M          = M
        self.K          = K
        # add bias node to input
        self.X          = hstack( (X, ones(( X.shape[0], 1 ) ) ) )
        self.targets    = t
        # weights; hidden and output
        wh              = random.rand(D, M) - 1/2
        wo              = random.rand(M, K) - 1/2

        self.layers     = [wh, wo]
        # activation functions:
        self.func       = lambda x: tanh(x)
        self.dfunc      = lambda x: 1 - x**2

    def forwardSingle(self, xi):
        ''' Performs a single forward pass in the network'''
        layerOutputs = [ []  for j in self.layers ]
        #forward pass
        a = xi.dot(self.layers[0])
        z = self.func(a)
        y = z.dot(self.layers[1])
        
        # save output
        layerOutputs[0].append(z);
        layerOutputs[1].append(y)
        return layerOutputs
    def backwardsSingle(self, ti, xi, forwardPass):
        '''Backprop + update of weights'''
        # prediction error
        dk = forwardPass[-1][0] - ti
        squaredError = dk**2
#         print(squaredError)
        # compute hidden activation; note elementwise product!!
        dj =         self.dfunc(forwardPass[0][0]) * (dk.dot(self.layers[-1].T))

        # update the weights
        E1 = forwardPass[0][0].T.dot(dk)
        E2 = xi.T.dot(dj)
        
        # update weights of layers
        self.layers[-1] -=         self.eta * E1 + self.gamma * self.layers[-1]
        
        self.layers[0]  -=         self.eta * E2 + self.gamma * self.layers[0]
        return squaredError
    def train(self, num):
        num   = int(num) # for scientific notation
        SSE   = zeros(num) # sum squared error
        preds = zeros((num, len(self.targets))) # predictions per run
        for iter in range(num):
            error = 0
            for idx, (ti, xi) in enumerate(zip(self.targets, self.X)):
                xi = array(xi, ndmin = 2)

                forwardPass = self.forwardSingle(xi)
                error += self.backwardsSingle(ti, xi, forwardPass)
                preds[iter, idx] = forwardPass[-1][0]

            SSE[iter] = .5 * error
        return SSE, preds

# perform a single forward pass and show the results
model = mlp(X, targets)
# perform a single pass
preds = array([              model.forwardSingle(              array(hstack( ( xi, 1) ),                    ndmin = 2))[-1]              for xi in X]).flatten()
# plot the results
fig, ax = subplots(subplot_kw  = {'projection': '3d'})
ax.scatter(x, y, preds.reshape(x.shape))
ax.set_xlabel('$x_1$', labelpad = 20)
ax.set_ylabel('$x_2$', labelpad = 20)
ax.set_zlabel('pdf', labelpad =20)
ax.set_title('Network output without training')
sb.set_context('poster')
show()


# # 3

# In[3]:

num = int(5e2) + 1
model = mlp(X, targets)
SSE, preds = model.train(num)
# plot every 250 cycles
cycleRange = arange(0,  num, num // 5)

# use at most 3 rows
nRows = 3
nCols = int(ceil(len(cycleRange)/nRows))
nCols = 2

fig, axes  = subplots(nRows,                      nCols,                      subplot_kw = {'projection': '3d'})


for axi, i in enumerate(cycleRange):
    ax = axes.flatten()[axi]
    ax.plot_surface(                    x,                    y,                    preds[i,:].reshape(x.shape),                    cstride = 1,                    rstride = 1)
    # formatting of plot
    ax.set_xlabel('$x_1$', labelpad = 20)
    ax.set_ylabel('$x_2$', labelpad = 20)
    ax.set_zlabel('pdf', labelpad =20)
    ax.set_title('cycles = {0}'.format(i))
    sb.set_style('white')
fig.delaxes(axes.flatten()[-1])
fig.tight_layout()
fig.suptitle('')
show()


# # 4

# In[4]:

# shuffle the indices
idx = random.permutation(len(targets))

# shuffle the data
shuffleX = X[idx,:]
# shuffle the targets as well (same indices)
shuffleTargets = targets[idx, :]
shuffleModel = mlp(shuffleX, shuffleTargets)
shuffleSSE, shufflePreds = shuffleModel.train(num)

fig, ax = subplots()
ax.plot( range(num),SSE,        range(num), shuffleSSE)

ax.set_xlabel('Iterations')
ax.set_ylabel('Sum squared error')
ax.legend(['shuffled','non-shuffled'],loc = 0)
fig.suptitle('Training error')
# savefig('../Figures/2.4.png')
show()


# Since the grid is linearly spaced, this will mean that nearby points will yield the same gradient in the error. This 'local' correlation will yield that the algorithm will change the weights in similar direction for a while, hence shuffling the data removes this 'local' correlation structure, yielding more likely to move in the different directions, constraining the algorithm, yielding faster convergence.

# # 5 

# In[2]:

# load the data
X, Y, target = list(np.loadtxt('../Data/a017_NNpdfGaussMix.txt').T)
tmp = int(np.sqrt(target.shape[0]))
# convert in shape for it to be plottable 
x = X.reshape(tmp,tmp)

y = Y.reshape(tmp, tmp)
# stack to create input data
X = np.vstack((X,Y)).T

# target vector
target = np.array(target, ndmin = 2).T

fig, ax = subplots(1,1, subplot_kw = {'projection': '3d'})
ax.plot_surface(x, y, target.reshape(tmp, tmp))
ax.set_xlabel('$x_1$', labelpad = 20)
ax.set_ylabel('$x_2$', labelpad = 20)
ax.set_zlabel('pdf', labelpad =20)
show()


# In[3]:

#randomly permute indices
idx = np.random.permutation(range(len(target)))
# keep track of the changes
shuffX = X[idx,:]; shuffTarget = target[idx]
# run mlp
model = mlp(shuffX, shuffTarget,                    eta = 1e-3,                    M = 40)

num = int(5e3) + 1
errors, preds = model.train(num = num)
# map back to original space
orgIdx = argsort(idx)
# get final prediction
finalPred = preds[-1, orgIdx]

# plot the final prediction and the target distribution
fig, ax = subplots(subplot_kw = {'projection': '3d'})
ax.scatter(x, y, target.reshape(x.shape), label = 'target')
ax.scatter(x, y, finalPred.reshape(x.shape), label = 'estimation')
ax.set_xlabel('$x_1$', labelpad = 20)
ax.set_ylabel('$x_2$', labelpad = 20)
ax.set_zlabel('pdf', labelpad = 20)
ax.legend(loc = 0)

fig, ax = subplots()
ax.plot(errors)
ax.set_xlabel('iterations')
ax.set_ylabel('Sum squared error')
ax.set_title('Training error')
show()


# One might improve the performance by adding more hidden nodes to the network. The hidden nodes essentially represent the degrees of freedom in the model. Increasing the number of hidden nodes might increase the fit on a trainingset, however it will also increase the modelling noise (i.e overfitting). Improvements might also be found in presenting the inputs / targets in a different feature space. Multi-layered perceptrons are notorious for being sensitive to how the data is represented. Another might be instead of taking a global learning rate, is make it adaptive (see conjugate gradient descent, momentum etc). 

# # 8

# We are using python hence netlab toolbox is not available to us. We opted for the neurolab toolbox which has a conjugate gradient method. The same parameters were used as in 7 to improve comparison.

# In[ ]:

import neurolab as nl

# inputs and targets
inp = shuffX
tar = shuffTarget

# specify same network structure as we have
# i.e. M = 40, D = 3, K = 1
# this function takes min/max of input space

# Specify activation functions;
# input - to hidden  is hyperbolic tangent
# hidden- to out is linear
tranfs = [nl.trans.TanSig(), nl.trans.PureLin()]
net = nl.net.newff(                   [ [np.min(X), np.max(X)] ] * inp.shape[1],                   [40, 1],                   transf= tranfs,                  )
# use conjugate gradient as method
net.trainf = nl.train.train_ncg # conjugate gradient
net.errorf  = nl.error.SSE()    # same as above

# init weight matrix between -.5,.5
for l in net.layers:
    l.initf = nl.init.InitRand([-.5, .5], 'wb')
net.init()

# Train network
errorNL = net.train(inp, tar, epochs=int(5e3),show = 1000)

# Simulate network
outNL = net.sim(inp)
outNL = outNL[argsort(idx)]

print(mean(errorNL), mean(errors[:10]))


# In[138]:

print(outNL.shape, x.shape)
# plot the performance versus our algorithm
# plot the final prediction and the target distribution
fig, ax = subplots(subplot_kw = {'projection': '3d'})
ax.scatter(x, y,           target.reshape(x.shape), label = 'target')
ax.scatter(x, y,           outNL.reshape(x.shape), label = 'estimation')

ax.set_xlabel('$x_1$', labelpad = 20)
ax.set_ylabel('$x_2$', labelpad = 20)
ax.set_zlabel('pdf', labelpad = 20)
ax.legend(loc = 0)
fig.suptitle('Sklearn prediction')
print('final SSE neruolab :\n {0}'.format(errorSklearn))
print('final SSE our alogorithm:\n{0}'.format(errors[-1]))
show()

