X = T.matrix("X")
y = T.matrix("y")
theta = T.matrix("theta")

h_theta = T.exp(-T.dot(X, theta))
fn_h_theta = theano.function([X, theta], h_theta)
loss_log_regr = - T.mean(y * T.log(1 / (1 + h_theta)) + (1 - y) * T.log(1 - 1 / (1 + h_theta)))

grad_loss_log_regr = T.grad(loss_log_regr, theta)

fn_grad_loss_log_regr = theano.function([X, theta, y], grad_loss_log_regr)

fn_loss_log_regr = theano.function([X, theta, y], loss_log_regr)([[1, 1], [-1, -1]], [[1], [1]], [[0], [1]])

theta_val = [[0], [0]]
X_val = [[1, 1], [-1, -1]]
y_val = [[0], [1]]

#X_batches = [[X_val[0]], [X_val[1]]]
#y_batches = [[[0]], [[1]]]

theta_val = [[2], [0]]
X_batches = [X_val]
y_batches = [y_val]

for i in range(100000):
    for X_val, y_val in zip(X_batches, y_batches):
        direction = fn_grad_loss_log_regr(X_val, theta_val, y_val)
        #print(direction)
        theta_val = theta_val - 0.01 * direction

print(theta_val, direction, fn_loss_log_regr(X_val, theta_val, y_val))

theta_val = [[0], [0]]
X_val = [[1, 1], [-1, -1], [1, 2]]
y_val = [[0], [1], [1]]

for i in range(100000):
  direction = fn_grad_loss_log_regr(X_val, theta_val, y_val)
  theta_val = theta_val - 0.01 * direction


print(theta_val, direction, fn_loss_log_regr(X_val, theta_val, y_val))

import numpy as np
theta_val = [[00], [10]]
X_val = np.concatenate([np.random.rand(10, 2), -np.random.rand(10, 2)])
#[[1, 1], [-1, -1], [1, 2]]
y_val = np.concatenate([np.zeros((10, 1)), np.ones((10, 1))])

theta_val = [[0], [10]]
for i in range(10000):
  direction = fn_grad_loss_log_regr(X_val, theta_val, y_val)
  theta_val = theta_val - 0.01 * direction

print(theta_val, direction, fn_loss_log_regr(X_val, theta_val, y_val),  fn_h_theta(X_val, theta_val))
