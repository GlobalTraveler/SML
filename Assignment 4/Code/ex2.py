import numpy as np
import seaborn as sb
from pylab import *
from scipy.stats import multivariate_normal as mv
from mpl_toolkits.mplot3d import Axes3D

'''
Goal: estimating multinomial gaussian using perceptron
'''
def mlp(X, t, eta = 1e-1, gamma = 0, num = 1e3, K = 1, M = 8):
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
    M = M #+ 1
    # input dimension; ass bias node
    D = X.shape[1] + 1
    # stack bias constant signal of ones
    X = np.hstack((X, np.ones((X.shape[0], 1))))
    # init weights
    # hidden weights and outputweights
    wh = np.random.rand(D, M) - 1/2
    wo = np.random.rand(M, K) - 1/2
    for i in range(num):
        # forward pass
        a = X.dot(wh)
        z = np.tanh(a)
        # compute prediction
        preds = z.dot(wo)
        # backward pass
        dk = t - preds
        # compute hidden activation; note elementwise product!!
        dj = (1 - z**2) * ((wo.dot(dk.T).T))
        # dj = z * (1 - z ) * ((wo.dot(dk.T)).T)
        # update th e weights
        E1 = z.T.dot(dk)
        E2 = X.T.dot(dj)
        wo += eta * E1 + gamma * wo
        wh += eta * E2 + gamma * wh
        error = np.sum((preds-t)**2)
        # print(error)
        if error < 1e-1:
            print('error low enough')
            break
        if error == nan:
            print('nan encountered')
            break
        # print(i)
    return preds, error


# q1
# create the target distribution
tmp = np.arange(-2,2, .1)
x, y = np.meshgrid(tmp, tmp)

mu = [0,0]
sigma = np.eye(len(mu)) * 2/5
dist = mv(mu, sigma)

X = np.vstack((x.flatten(), y.flatten()))
# Y =  np.log(dist.pdf(X.T).reshape(x.shape))* 3
Y = dist.pdf(X.T).reshape(x.shape)  * 3
# shufIdx = np.random.permutation(range(X.shape[0]))
# shufX = X[shufIdx,:]
# shufT = targets[shufIdx]
targets = np.array(Y.flatten(), ndmin = 2).T

# dist = mv(0, 2/5)
# tmp = np.linspace(-1,1, 100)
# targets = np.array(dist.pdf(tmp), ndmin = 2).T  * 3
# targets = np.array(np.sin(tmp),ndmin = 2).T
# targets = np.array(tmp**2,ndmin = 2).T
# targets = np.array( np.exp(-(tmp)**2), ndmin = 2).T
# X = np.array(tmp, ndmin = 2).T
idx = np.random.permutation(range(len(targets)))
X = X.T
# X = X[idx,:]
# targets = targets[idx]
# print(idx)
preds, error = mlp(X, targets, eta = 1e-3,M = 8, num = 1800)
fig, ax = subplots(1,1, subplot_kw = {'projection': '3d'})
# ax.scatter(X, targets)
# ax.scatter(X, preds)
# ax.legend(['target','prediction'], loc = 0)
# ax.set_title('SSE =' + str(error))
ax.scatter(xs = X[:,0], ys = X[:,1], zs = preds)
ax.scatter(xs = X[:,0], ys = X[:,1], zs = targets)

ax.legend(['prediction', 'target'], loc = 0)

fig, ax = subplots(1,1, subplot_kw = {'projection': '3d'})
ax.plot_surface(X[:,0].reshape(x.shape), X[:,1].reshape(x.shape), preds.flatten().reshape(x.shape))
show()

# # test case: this works
# X = np.array([[0,0],[0,1],[1,0],[1,1]])
# targets = np.array([0,1,1,0], ndmin =2 ).T
# pred, error = mlp(X, targets, eta = .1)
# print(error)


close('all')
# fig, ax = subplots(1, 1, subplot_kw = {'projection': '3d'})
# x, y = list(X.T)
#
# tmp = np.argsort(X,0)
# tmpx, tmpy = tmp[:,0], tmp[:,1]
# d = int(np.sqrt(x.shape[0]))
# shape = (d,d)
# # x, y, preds = x[tmpx].reshape(shape), y[tmpy].reshape(shape),  preds[tmpx].flatten().reshape(shape)
# ax.scatter(xs = x, ys= y, zs = preds)
# ax.scatter(xs = x, ys = y, zs = targets)
# show(0)
