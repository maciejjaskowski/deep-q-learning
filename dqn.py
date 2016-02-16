# Alt + Shift + E

from __future__ import print_function
from __future__ import division

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


class ReplayMemory(object):
    def __init__(self, size=1000000, grace=10000):
        self.max_size = size
        self.grace = grace
        self.list = []

    def init_state(self, s0):
        if len(self.list) > 0:
            self.list[-1] = s0
        else:
            self.list.append(s0)
        try:
            assert len(self.list) % 4 == 1
        except:
            print("wtf")

    def append(self, a0, r0, fri, s1):
        el = [a0, r0, fri, s1]

        self.list.extend(el)
        if len(self) > self.max_size + self.grace * 4:
            self.list = self.list[self.grace * 4:]
        try:
            assert len(self.list) % 4 == 1
        except:
            print("wtf")

    def sample(self, sample_size):
        import random
        indices = random.sample(xrange(len(self)), sample_size)
        return [self[i] for i in indices]

    def __len__(self):
        return int((len(self.list) - 1) / 4)

    def __getitem__(self, idx):
        return tuple(self.list[idx*4:idx*4+5])


class DQNAlgo:
    def __init__(self, n_actions, replay_memory, initial_weights_file=None):
        self.ignore_feedback = False
        self.alpha = 0.00025
        # update frequency ?
        # gradient momentum ? 0.95
        # squared gradient momentum ? 0.95
        # min squared gradient ? 0.01
        self.save_every_n_frames = 100000 # ~ once per hour

        self.final_exploration_frame = 1000000
        self.replay_start_size = 50000
        self.i_frames = 0

        self.state = None
        self.initial_epsilon = 1
        self.final_epsilon = 0.1
        self.epsilon = self.initial_epsilon
        self.gamma = 0.99
        self.replay_memory = replay_memory

        self.minibatch_size = 32
        #self.replay_memory_size = 1000000

        self.target_network_update_frequency = 10000

        s0_var, a0_var, r0_var, s1_var, future_reward_indicator_var = T.tensor4("s0", dtype=theano.config.floatX), T.bmatrix("a0"), T.wcol(
            "r0"), T.tensor4("s1", dtype=theano.config.floatX), T.bcol(
            "future_reward_indicator")
        self.n_actions = n_actions
        self.a_lookup = np.eye(self.n_actions, dtype=np.int8)

        self.network = build_cnn(n_actions=self.n_actions, input_var=s0_var)
        print("Compiling forward.")
        self.forward = theano.function([s0_var], lasagne.layers.get_output(self.network, deterministic=True))

        self.network_stale = build_cnn(n_actions=self.n_actions, input_var=s1_var)
        print("Compiling forward stale.")
        self.forward_stale = theano.function([s1_var],
                                             lasagne.layers.get_output(self.network_stale, deterministic=True))

        if initial_weights_file is not None:
            with np.load(initial_weights_file) as initial_weights:
                param_values = [initial_weights['arr_%d' % i] for i in range(len(initial_weights.files))]
                lasagne.layers.set_all_param_values(self.network, param_values)

        self._update_network_stale()

        self.loss, self.err = build_loss(self.network, self.network_stale,
                               (s0_var, a0_var, r0_var, s1_var, future_reward_indicator_var), self.gamma)

        params = lasagne.layers.get_all_params(self.network, trainable=True)
        updates = lasagne.updates.rmsprop(self.loss, params, learning_rate=1.0, rho=0.95,
                                          epsilon=1e-6)  # TODO RMSPROP in the paper has slightly different definition (see Lua)
        print("Compiling train_fn.")
        self.train_fn = theano.function([s0_var, a0_var, r0_var, s1_var, future_reward_indicator_var],
                                        [self.loss, self.err], updates=updates)
        print("Compiling loss_fn.")
        self.loss_fn = theano.function([s0_var, a0_var, r0_var, s1_var, future_reward_indicator_var],
                                        self.loss)

    def init_state(self, state):
        self.state = self._prep_state(state)
        self.replay_memory.init_state(self.state)

    def _update_network_stale(self):
        lasagne.layers.set_all_param_values(self.network_stale, lasagne.layers.get_all_param_values(self.network))

    @staticmethod
    def _prep_state(state):
        return np.reshape(np.stack(state, axis=0), (1, 4, 80, 80))

    def action(self):
        import random
        if self.i_frames < self.final_exploration_frame:
            if self.i_frames % 10000 == 50:
                self.epsilon = (self.final_epsilon - self.initial_epsilon) * (self.i_frames / self.final_exploration_frame) + self.initial_epsilon
                print("epsilon: ", self.epsilon)
        else:
            self.epsilon = self.final_epsilon

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
        if self.ignore_feedback:
            return

        self.replay_memory.append(self.a_lookup[exp.a0], min(1, max(-1, exp.r0)), 1-int(exp.game_over), self._prep_state(exp.s1))

        import random

        if len(self.replay_memory) > self.replay_start_size:
            sample = zip(*self.replay_memory.sample(self.minibatch_size))
            # print(self.replay_memory.list)
            # print("###########")
            try:
                    s0 = np.array(sample[0], dtype=theano.config.floatX).reshape(self.minibatch_size, 4, 80, 80)
            except:
                print(sample[0])

            a0 = np.array(sample[1], dtype=np.int8).reshape(self.minibatch_size, self.n_actions)

            r0 = np.array(sample[2], dtype=np.int16).reshape(self.minibatch_size, 1)

            future_reward_indicators = np.array(sample[3], dtype=np.int8).reshape(self.minibatch_size, 1)

            s1 = np.array(sample[4], dtype=theano.config.floatX).reshape(self.minibatch_size, 4, 80, 80)

            t = self.train_fn(s0, a0, r0, s1, future_reward_indicators)
            print('loss: ', t)

            if self.i_frames % self.target_network_update_frequency == 0:
                self._update_network_stale()

        if self.i_frames % self.save_every_n_frames == 100:  # 30 processed frames / s
            filename = 'weights_' + str(self.i_frames) + '.npz'
            print("File saved: ", filename)
            np.savez(filename, *lasagne.layers.get_all_param_values(self.network))



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
    qs.tag.test_value = np.random.rand(32, 6).astype(dtype=theano.config.floatX)
    y = r0_var + gamma * future_reward_indicator_var * T.max(qs, axis=1)  # 32 x 1
    #y = r0_var + gamma * future_reward_indicator_var * qs[:,0]  # 32 x 1

    out = lasagne.layers.get_output(network, deterministic=True)  # 32 x 6
    out.tag.test_value = np.random.rand(1, 6).astype(dtype=theano.config.floatX)
    q = T.sum(T.dot(a0_var, T.transpose(out)), axis=1)  # 32 x 1
    err = y - q
    #err = T.max(T.stack(T.neg(T.ones_like(err)), T.min(T.stack(T.ones_like(err), err), axis=0)), axis=0) # cap with -1 and 1 elementwise
    loss = err ** 2
    return loss.mean(), (y-q).mean()  # TODO or sum? -> alpha depends on that.
