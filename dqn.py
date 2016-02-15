# Alt + Shift + E

from __future__ import print_function

import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T

import lasagne


# ################## Download and prepare the MNIST dataset ##################
# This is just some way of getting the MNIST dataset from an online location
# and loading it into numpy arrays. It doesn't involve Lasagne at all.

def load_dataset():
    # We first define a download function, supporting both Python 2 and 3.
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve

    def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
        print("Downloading %s" % filename)
        urlretrieve(source + filename, filename)

    # We then define functions for loading MNIST images and labels.
    # For convenience, they also download the requested files if needed.
    import gzip

    def load_mnist_images(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the inputs in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        # The inputs are vectors now, we reshape them to monochrome 2D images,
        # following the shape convention: (examples, channels, rows, columns)
        data = data.reshape(-1, 1, 28, 28)
        # The inputs come as bytes, we convert them to float32 in range [0,1].
        # (Actually to range [0, 255/256], for compatibility to the version
        # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
        return data / np.float32(256)

    def load_mnist_labels(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the labels in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        # The labels are vectors of integers now, that's exactly what we want.
        return data

    # We can now download and read the training and test set images and labels.
    X_train = load_mnist_images('train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

    # We reserve the last 10000 training examples for validation.
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
    return X_train, y_train, X_val, y_val, X_test, y_test


def build_cnn(n_actions, input_var=None):
    network = lasagne.layers.InputLayer(shape=(None, 4, 80, 80),
                                        input_var=input_var)

    network = lasagne.layers.Conv2DLayer(
        network, num_filters=16, filter_size=(8, 8), stride=4,
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform())

    network = lasagne.layers.Conv2DLayer(
        network, num_filters=32, filter_size=(4, 4), stride=2,
        nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers.DenseLayer(
        network,
        num_units=256,
        nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers.DenseLayer(
        network,
        num_units=n_actions)

    return network


# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# them to GPU at once for slightly improved performance. This would involve
# several changes in the main program, though, and is not demonstrated here.

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


# ############################## Main program ################################
# Everything else will be handled in our main program now. We could pull out
# more functions to better separate the code, but it wouldn't make it any
# easier to read.

def main(num_epochs=500):
    # Load the dataset
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    T.log

    # Create neural network model (depending on first command line parameter)
    print("Building model and compiling functions...")

    network = build_cnn(input_var)

    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):

    prediction = lasagne.layers.get_output(network)  # Q(s) \in R^4

    # a = argmax(prediction) -> Emulator
    # [0 -> nothing, 1 -> left, 2 -> right, 3 -> shoot]

    # Emulator -> r, s'
    gamma = 0.9

    loss = (r + gamma * max_a_prime(Q_prev(s_prime, a_prime)) - Q_now(s, a)) ** 2

    # a' -> Emulator


    # We could add some weight decay as well here, see lasagne.regularization.

    params = lasagne.layers.get_all_params(network, trainable=True)

    updates = lasagne.updates.sgd(loss, params, learning_rate=0.01)

    # As a bonus, also create an expression for the classification accuracy:
    # test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
    #                  dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a number of training games

        start_time = time.time()
        average_train_reward, average_train_loss = dqnTeacher.teach(50)
        # train_err = 0
        # train_batches = 0
        # start_time = time.time()
        # for batch in iterate_minibatches(X_train, y_train, 500, shuffle=True):
        #     inputs, targets = batch
        #     train_err += train_fn(inputs, targets)
        #     train_batches += 1

        # And 3 test games
        old_epsilon = dqn.epsilon
        dqn.epsilon = 0.01
        average_test_reward, average_test_loss = dqnTeacher.teach(3)
        dqn.epsilon = old_epsilon

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(average_train_reward))
        print("  validation loss:\t\t{:.6f}".format(average_test_reward))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))

    # After training, we compute and print the test error:
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))

    # Optionally, you could now dump the network weights to a file like this:
    # np.savez('model.npz', *lasagne.layers.get_all_param_values(network))
    #
    # And load them again later on like this:
    # with np.load('model.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(network, param_values)


class DQNAlgo:
    def __init__(self, n_actions):
        self.alpha = 0.00025
        # update frequency ?
        # gradient momentum ? 0.95
        # squared gradient momentum ? 0.95
        # min squared gradient ? 0.01
        self.initial_epsilon = 1
        self.final_epsilon = 0.1
        self.final_exploration_frame = 1000000
        self.replay_start_size = 50000
        self.i_frames = 0

        self.state = None
        self.epsilon = 0.05
        self.gamma = 0.99
        self.replay_memory = []

        self.minibatch_size = 32
        self.replay_memory_size = 1000000

        self.target_network_update_frequency = 10000

        s0_var, a0_var, r0_var, s1_var, future_reward_indicator_var = T.dtensor4("s0"), T.bmatrix("a0"), T.wcol(
            "r0"), T.dtensor4("s1"), T.bcol(
            "future_reward_indicator")  # TODO: dtensor4 would work but can we influence the convolutional layer to make it work with ftensor4? ?
        self.n_actions = n_actions

        self.network = build_cnn(n_actions=self.n_actions, input_var=s0_var)
        self.forward = theano.function([s0_var], lasagne.layers.get_output(self.network, deterministic=True))

        self.network_stale = build_cnn(n_actions=self.n_actions, input_var=s1_var)
        self.forward_stale = theano.function([s1_var],
                                             lasagne.layers.get_output(self.network_stale, deterministic=True))
        self._update_network_stale()

        self.loss = build_loss(self.network, self.network_stale,
                               (s0_var, a0_var, r0_var, s1_var, future_reward_indicator_var), self.gamma)

        params = lasagne.layers.get_all_params(self.network, trainable=True)
        updates = lasagne.updates.rmsprop(self.loss, params, learning_rate=1.0, rho=0.95,
                                          epsilon=1e-6)  # TODO RMSPROP in the paper has slightly different definition (see Lua)

        self.train_fn = theano.function([s0_var, a0_var, r0_var, s1_var, future_reward_indicator_var],
                                        self.loss, updates=updates)

    def init_state(self, state):
        self.state = self._prep_state(state)

    def _update_network_stale(self):
        lasagne.layers.set_all_param_values(self.network_stale, lasagne.layers.get_all_param_values(self.network))

    @staticmethod
    def _prep_state(state):
        return np.reshape(np.stack(state, axis=0), (1, 4, 80, 80))

    def action(self):
        import random
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        else:
            return self.best_action()

    def best_action(self):
        q = self.forward(self.state)
        return np.argmax(q)

    def feedback(self, exp):
        # exp -> s0 a0 r0 s1 game_over
        self.i_frames += 1

        if self.i_frames < 5:
            self.a_lookup = np.eye(self.n_actions, dtype=np.int8)

        self.replay_memory.append((self._prep_state(exp.s0), self.a_lookup[exp.a0], exp.r0, self._prep_state(exp.s1), 1 - int(exp.game_over)))
        if len(self.replay_memory) > self.replay_memory_size + 10000:
            self.replay_memory = self.replay_memory[10000:]
        import random

        if len(self.replay_memory) > self.replay_start_size:
            sample = zip(*random.sample(self.replay_memory, self.minibatch_size))
            s0 = np.array(sample[0], dtype=np.float32).reshape(self.minibatch_size, 4, 80, 80)

            a0 = np.array(sample[1], dtype=np.int8).reshape(self.minibatch_size, self.n_actions)

            r0 = np.array(sample[2], dtype=np.int16).reshape(self.minibatch_size, 1)

            s1 = np.array(sample[3], dtype=np.float32).reshape(self.minibatch_size, 4, 80, 80)

            future_reward_indicators = np.array(sample[4], dtype=np.int8).reshape(self.minibatch_size, 1)

            self.train_fn(s0, a0, r0, s1, future_reward_indicators)

            if self.i_frames % self.target_network_update_frequency == 0:
                self._update_network_stale()


def build_loss(network, network_stale, exp_var, gamma):
    s0_var, a0_var, r0_var, s1_var, future_reward_indicator_var = exp_var
    # s0_var mini_batch x 4 x,80 x 80
    # a0_var mini_batch x 1,
    # r0_var mini_batch x 1,
    # s1_mini_batch x 4 x 80 x 80
    future_reward_indicator_var.tag.test_value = np.random.rand(32, 1).astype(dtype=np.int8)
    r0_var.tag.test_value = np.random.rand(32, 1).astype(dtype=np.int16)
    a0_var.tag.test_value = np.random.rand(32, 6).astype(dtype=np.int8)

    qs = lasagne.layers.get_output(network_stale, deterministic=True)  # 32 x 6
    qs.tag.test_value = np.random.rand(32, 6).astype(dtype=np.float32)
    y = r0_var + gamma * future_reward_indicator_var * T.max(qs, axis=1)  # 32 x 1

    out = lasagne.layers.get_output(network, deterministic=True)  # 32 x 6
    out.tag.test_value = np.random.rand(1, 6).astype(dtype=np.float32)
    q = T.sum(T.dot(a0_var, T.transpose(out)), axis=1)  # 32 x 1
    loss = (y - q) ** 2
    return loss.mean()  # TODO or sum? -> alpha depends on that.
