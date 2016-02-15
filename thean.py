import theano
from theano import tensor as T
import numpy

rng = numpy.random

X = T.matrix("X")
y = T.matrix("y")
theta = theano.shared(rng.randn(2,1), name="theta")

h_theta = T.exp(-T.dot(X, theta))
fn_h_theta = theano.function([X], h_theta)
loss_log_regr = - T.mean(y * T.log(1 / (1 + h_theta)) + (1 - y) * T.log(1 - 1 / (1 + h_theta)))

grad_loss_log_regr = T.grad(loss_log_regr, theta)

fn_grad_loss_log_regr = theano.function([X, y], grad_loss_log_regr)

fn_loss_log_regr = theano.function([X, y], loss_log_regr)([[1, 1], [-1, -1]], [[0], [1]])

train = theano.function([X, y], outputs=[loss_log_regr, theta], updates=[(theta, theta - 0.01 * grad_loss_log_regr)])

import numpy as np

theta_val = [[00], [10]]
X_val = np.concatenate([np.random.rand(10, 2), -np.random.rand(10, 2)])
y_val = np.concatenate([np.zeros((10, 1)), np.ones((10, 1))])


for i in range(10000):
    print(train(X_val, y_val))

print(theta.get_value())
