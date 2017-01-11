import numpy as np
import seaborn as sb
from pylab import *
from scipy.stats import multivariate_normal as mv
from mpl_toolkits.mplot3d import Axes3D

'''
Goal: estimating multinomial gaussian using perceptron
'''
def mlp(X, t, eta = 1e-3, gamma = 0, num = 1e3, K = 1, M = 8, save = 4):
    '''
    Multi-layered-pereptron
    K = output nodes
    M = hidden nodes
    Assumes the input data X is samples x feature dimension
    Returns:
        prediction and error
    '''
    num = int(num)
    # add bias node; note the bias is absorbed in the weights of the
    # second layer ()
    M = M + 1
    # input dimension; ass bias node
    D = X.shape[1] + 1
    # stack bias constant signal of ones
    X = np.hstack((X, np.ones((X.shape[0], 1))))
    # init weights
    # hidden weights and outputweights
    wh = np.random.rand(D, M) - 1/2
    wo = np.random.rand(M, K) - 1/2
    errors = np.zeros(num + 1)
    preds = [[]] * (save + 1)
    idx = 0
    tmp = num // save
    for i in range(num + 1):
        # forward pass
        a = X.dot(wh);  z = np.tanh(a);   y = z.dot(wo)
        # backward pass
        dk = y - t
        # compute hidden activation; note elementwise product!!
        dj = (1 - z**2) * ((wo.dot(dk.T).T))
        # dj = z * (1 - z ) * ((wo.dot(dk.T)).T)
        # update th e weights
        E1 = z.T.dot(dk); E2 = X.T.dot(dj)
        wo -= eta * E1 + gamma * wo
        wh -= eta * E2 + gamma * wh
        error = np.sum((y-t)**2)
        errors[i] = error
        # print(error)
        if error < 1e-1:
            print('error low enough')
            break
        if error == nan:
            print('nan encountered')
            break
        if  i % tmp == 0:
            print('pass', i)
            preds[idx] = [y,i]
            idx += 1
        # print(i)
    return errors, preds

def plot_results(errors, pred, tar, inputs):
    # plot the end result
    tmp = int(np.sqrt(inputs.shape[0]))
    print(inputs.shape)
    inputs = inputs.reshape(tmp, tmp, inputs.shape[1])
    x = inputs[:,:,0]; y = inputs[:,:,1]

    fig, ax = subplots(1,1, subplot_kw = {'projection': '3d'})
    ax.scatter(x, y, preds[-1][0].reshape(x.shape))
    ax.scatter(x, y, tar.reshape(x.shape))
    ax.legend(['prediction', 'target'], loc = 0)
    savefig('../Figures/ex231')

    # plot the training
    fig, ax = subplots(2,2, subplot_kw = {'projection': '3d'})
    for idx, axi in enumerate(ax.flatten()):
        data, cycle = preds[idx]
        axi.scatter(x, y, data.flatten().reshape(x.shape))
        axi.set_title('cycle : {0}'.format(cycle))
        axi.set_xlabel('$x_1$', labelpad = 10)
        axi.set_ylabel('$x_2$', labelpad = 10)
        axi.set_zlabel('pdf', labelpad = 10 )
    fig.tight_layout()
    savefig('../Figures/232')

    # plot the errors
    fig, ax = subplots(1,1)
    ax.plot(errors)
    ax.set_ylabel('SSE')
    ax.set_xlabel('Iteration')
    ax.set_title('Sum squared error as a function of iteration')

    show(0)
close('all')
# q1
# create the target distribution
tmp = np.arange(-2, 2, .2)
x, y = np.meshgrid(tmp, tmp)

mu = [0,0]
sigma = np.eye(len(mu)) * 2/5
dist = mv(mu, sigma)

X = np.vstack((x.flatten(), y.flatten())).T
Y = dist.pdf(X).reshape(x.shape)  * 3
targets = np.array(Y.flatten(), ndmin = 2).T


# print(targets.shape); assert 0
errors, preds = mlp(X, targets, num = 1000)
plot_results(errors, preds, targets, X)
print(errors[-1])
#  q 4 : random shuffle index
idx = np.random.permutation(range(len(targets)))
shuffleX = X[idx,:]
shuffleTargets = targets[idx]
errors_shuff, preds = mlp(shuffleX, shuffleTargets, num = 1000)
print(errors[-1])
plot_results(errors, preds, shuffleTargets, shuffleX)

# it is faster sometimes, dunno why;
# im asuming that it hits the right values such that the constraints get updated
# farely quickly
fig, ax = subplots(1,1)
ax.plot(errors, label = 'original')
ax.plot(errors_shuff, label = 'shuffled')
ax.legend(loc = 0)

# q5 : load file, create distribution
X, Y, target = list(np.loadtxt('../Data/a017_NNpdfGaussMix.txt').T)
tmp = int(np.sqrt(target.shape[0]))
x = X.reshape(tmp,tmp)
y = Y.reshape(tmp, tmp)

X = np.vstack((X,Y)).T
target = np.array(target, ndmin = 2).T
idx = np.random.permutation(range(len(target)))
X = X[idx,:]; target = target[idx]
# fig, ax = subplots(1,1, subplot_kw = {'projection': '3d'})
# ax.plot_surface(x, y, target.reshape(tmp, tmp), cmap = 'viridis')

# # q6 train the network on the dataset
# errors, preds = mlp(X, target, eta = 1e-4,num = 5 * 1e3,  M = 40)
# plot_results(errors, preds, target, X)
show(1)
