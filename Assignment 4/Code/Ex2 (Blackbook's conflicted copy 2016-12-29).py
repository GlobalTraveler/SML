import numpy as np
import seaborn as sb
from pylab import *
from scipy.stats import multivariate_normal as mv
from mpl_toolkits.mplot3d import Axes3D




class mlp(object):
    def __init__(self, M = 5, D = 2, K = 1, eta = .01,\
                func = 'tanh(x)', state = 'test', n_patterns = 100):
        '''
        Multi-layered perceptron with:
            D input nodes
            M hidden nodes
            K output nodes
            eta learning parameter
        '''
        # add bias
        # D = D + 1
        # lazy loading of activation function
        from sympy.parsing.sympy_parser import parse_expr,\
        standard_transformations, implicit_multiplication_application

        transformations = (standard_transformations + (implicit_multiplication_application,))
        func = parse_expr(func, transformations = transformations)

        from sympy import Derivative, lambdify, symbols
        params   = symbols('x')
        dfunc    = func.diff(params)


        # Activation function and derivative
        self.func   = lambdify(params,func, 'numpy')

        self.dfunc  = lambdify(params, dfunc, modules = 'numpy')
        # end of lazy loading
        # init weights
        # weights_in_hidden = np.random.randn(D, M)
        # weights_hidden_out = np.random.randn(M, K)
        weights_in_hidden = np.random.rand(D, M) * 1/2 - 1/2
        weights_hidden_out = np.random.rand(M,K) * 1/2 - 1/2

        if state == 'test':
            rand = np.random.randn
            self.patterns   = np.random.randint(0,2, (n_patterns, D))*2-1

            # make sure that the targets are linearly separable
            self.targets    = self.patterns.dot(\
            np.random.randn(D,M).dot(np.random.randn(M, K)))
            tmp = self.targets >= 0
            self.targets[tmp == 1] = 1
            self.targets[tmp == 0] = -1
        else:
            self.patterns, self.targets = state

        print(self.patterns.shape, self.targets.shape)

            # self.targets    = np.sign(self.targets)
        self.layers = [weights_in_hidden, weights_hidden_out]
        # learning rate
        self.eta = eta

    def forward_pass(self, inputs):
        output = inputs
        outputs = []

        first = self.func(inputs.dot(self.layers[0]))
        second = first.dot(self.layers[-1])
        outputs.append(first)
        outputs.append(second)

        # for weights in self.layers:
            # output = self.func( output.dot( weights) )
            # outputs.append(output)
        return outputs

    def back_prop(self, forward, targets):
        # keep upating the targets recursively
        grads =  []
        for idx in range(len(self.layers)):
            # layers and activation for a layer
            weights = self.layers[- idx - 1]
            activation = forward[- idx - 1]
            # compute the delta for this layer
            # if idx == 0 :
            delta = targets - activation
            # print(delta.shape)#
            # compute the gradient for the activation of layer
            grad = self.dfunc(activation)
            grads.append(weights)
            # print(delta.shape, weights.shape, grad.shape)
            # update the weights
            weights -=  self.eta * weights.dot(delta.T).dot(grad)
            # propogate the error to the previous layer
            targets = delta
        return grads

        # assert 0

    def train(self, threshold = .1, alpha = True, maxiter = int(1e4)):
        idx = 0; errors = []
        grads = [[]]*3
        # print(grads)
        while alpha:
            # forward pass
            outputs = self.forward_pass(self.patterns)
            # backward pass
            grads[0] = grads[1]
            grads[1] = grads[2]
            tmp  = self.back_prop(outputs, self.targets)

            grads[2] = tmp
            if any(np.isnan(grads[-1][0])) or any(np.isnan(grads[-1][1])):
                for i in grads:
                    print(i)
                assert 0

            # save errors
            errors.append(np.mean((self.targets - outputs[-1])**2))
            # if idx % 1000:
            #     # self.eta = np.random.randn() *2
            #     print(errors[-1])
            if idx > maxiter:
                alpha = False
                return errors, idx
            idx += 1

close('all')
# q1
tmp = np.arange(-2,2, .1)
x, y = np.meshgrid(tmp, tmp)

dist = mv([0, 0], np.eye(2) * 2/5. * 3 )

X = np.vstack((x.flatten(),y.flatten()))
Y =  dist.pdf(X.T).reshape(x.shape)
fig, ax = subplots(2,1, subplot_kw = {'projection': '3d'})


ax[0].plot_surface(x, y, Y, cmap = 'viridis')
# show()
# print(Y.shape); assert 0

tmp = state = [X.T, np.array(Y.flatten(), ndmin = 2).T]
net  = mlp(M = 8, eta = 1e-1, n_patterns  = 4, state = tmp)
errors, iters = net.train(threshold = 1e-1, maxiter = int(1000))
out = net.forward_pass(net.patterns)
# print(errors[-1])
print((out[-1] - net.targets)**2, iters)


# fig, ax = subplots(1,1, subplot_kw = {'projection': '3d'})
ax[1].plot_surface(x, y, out[-1].reshape(x.shape))
# print('>',out[-1].T,'\t', net.targets.T)
# fig, ax = subplots(1,2)
# ax[0].plot(range(iters+1), errors)
# ax[1].plot(net.targets, out[-1],'.')
# ax[1].set_ylim([-1,1])
# ax[1].set_xlim([-1,1])
sb.set_context('poster')
fig.tight_layout()
show(0)
