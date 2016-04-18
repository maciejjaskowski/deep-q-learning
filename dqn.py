# Alt + Shift + E

from __future__ import print_function
from __future__ import division

import numpy as np
import theano
import theano.tensor as T

import lasagne


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

    def append(self, a0, r0, fri, s1):
        el = [a0, r0, fri, s1]

        self.list.extend(el)
        if len(self) > self.max_size + self.grace * 4:
            self.list = self.list[self.grace * 4:]

    def sample(self, sample_size):
        import random
        indices = random.sample(xrange(len(self)), sample_size)
        return [self[i] for i in indices]

    def __len__(self):
        return int((len(self.list) - 1) / 4)

    def __getitem__(self, idx):
        return tuple(self.list[idx * 4:idx * 4 + 5])


class DQNAlgo:
    def __init__(self, n_actions, replay_memory, build_network, updates, screen_size, initial_weights_file=None):
        self.screen_size = screen_size
        self.mood_q = None
        self.last_q = 0
        self.n_parameter_updates = 0
        self.alpha = 0.00025
        # update frequency ?
        # gradient momentum ? 0.95
        # squared gradient momentum ? 0.95
        # min squared gradient ? 0.01
        self.save_every_n_frames = 100000  # ~ once per hour

        self.final_exploration_frame = 1000000
        self.replay_start_size = 50000
        self.i_action = 0

        self.state = None
        self.initial_epsilon = 1
        self.final_epsilon = 0.1
        self.epsilon = self.initial_epsilon
        self.gamma = 0.99
        self.replay_memory = replay_memory

        self.log_frequency = 1

        self.minibatch_size = 32
        # self.replay_memory_size = 1000000

        self.target_network_update_frequency = 10000

        s0_var = T.tensor4("s0", dtype=theano.config.floatX)
        a0_var = T.bmatrix("a0")
        r0_var = T.wcol("r0")
        s1_var = T.tensor4("s1", dtype=theano.config.floatX)
        future_reward_indicator_var = T.bcol("future_reward_indicator")

        self.n_actions = n_actions
        self.a_lookup = np.eye(self.n_actions, dtype=np.int8)

        self.network = build_network(n_actions=self.n_actions, input_var=T.cast(s0_var, 'float32') / np.float32(256),
                                     screen_size=self.screen_size)
        print("Compiling forward.")
        self.forward = theano.function([s0_var], lasagne.layers.get_output(self.network, deterministic=True))

        self.network_stale = build_network(n_actions=self.n_actions, input_var=T.cast(s1_var, 'float32') / np.float32(256),
                                           screen_size=self.screen_size)
        print("Compiling forward_stale.")
        self.forward_stale = theano.function([s1_var],
                                             lasagne.layers.get_output(self.network_stale, deterministic=True))

        if initial_weights_file is not None:
            with np.load(initial_weights_file) as initial_weights:
                param_values = [initial_weights['arr_%d' % i] for i in range(len(initial_weights.files))]
                lasagne.layers.set_all_param_values(self.network, param_values)
            self.i_action -= self.replay_start_size

        self._update_network_stale()

        out = lasagne.layers.get_output(self.network)
        out_stale = lasagne.layers.get_output(self.network_stale)
        self.loss, self.err, __y, __q = build_loss(out=out,
                                                   out_stale=out_stale,
                                                   a0_var=a0_var,
                                                   r0_var=r0_var,
                                                   future_reward_indicator_var=future_reward_indicator_var,
                                                   gamma=self.gamma)

        params = lasagne.layers.get_all_params(self.network, trainable=True)

        print("Compiling train_fn.")
        self.train_fn = theano.function([s0_var, a0_var, r0_var, s1_var, future_reward_indicator_var],
                                        [self.loss, self.err, T.transpose(__y), T.transpose(__q), out, out_stale],
                                        updates=updates(self.loss, params))
        print("Compiling loss_fn.")
        self.loss_fn = theano.function([s0_var, a0_var, r0_var, s1_var, future_reward_indicator_var],
                                       self.loss)

        self.test_mode = False

    def log(self, *args):
        if self.i_action % 10000 < self.log_frequency:
            print(self.i_action, *args)

    def init_state(self, state):
        self.state = self._prep_state(state)
        if self.replay_memory is not None:
            self.replay_memory.init_state(self.state)

    def _update_network_stale(self):
        print("{i_frame} | Updating stale network.".format(i_frame=self.i_action))
        lasagne.layers.set_all_param_values(self.network_stale, lasagne.layers.get_all_param_values(self.network))

    def _prep_state(self, state):
        return np.reshape(np.stack(state, axis=0), (1, 4, self.screen_size, self.screen_size))

    def action(self, _state):
        import random
        if self.test_mode:
            self.epsilon = 0.1
        else:

            if self.i_action < self.final_exploration_frame:
                if self.i_action % 10000 == 50:
                    self.epsilon = (self.final_epsilon - self.initial_epsilon) * (
                        self.i_action / self.final_exploration_frame) + self.initial_epsilon

            else:
                self.epsilon = self.final_epsilon
            self.log("{i_frame} | epsilon: {epsilon}".format(i_frame=self.i_action, epsilon=self.epsilon))

        if random.random() < self.epsilon:
            action = random.randint(0, self.n_actions - 1)
            #print("{i_frame} | random action: {action}".format(i_frame=self.i_frames, action=action))
            return action
        else:
            return self._best_action(self._prep_state(_state))

    def _best_action(self, state):
        q = self.forward(state)
        self.last_q = np.max(q)
        action = np.argmax(q)
#        print("{i_frame} | q: {q} | action: {action}".format(i_frame=self.i_frames, q=q, action=action))
        return action

    def feedback(self, exp):
        if self.test_mode:
            return
        # exp -> s0 a0 r0 s1 game_over
        self.i_action += 1
        self.state = self._prep_state(exp.s1)

        r0_clipped = min(1, max(-1, exp.r0))
        fri = 1 - int(exp.game_over)

        if self.mood_q:
            expectation = np.max(self.forward(self.state))
            surprise = (r0_clipped + self.gamma * expectation) - self.last_q
            self.mood_q.put({'i_frame': self.i_action, 'surprise': surprise, "expectations": expectation})

        if self.replay_memory is None:
            return

        # if exp.r0 != 0 or r0_clipped != 0:
        #     print("{i_frame} | reward: {reward} | creward: {creward}"
        #           .format(i_frame=self.i_action, reward=exp.r0, creward=r0_clipped))

        self.replay_memory.append(self.a_lookup[exp.a0], r0_clipped, fri, self.state)

        if len(self.replay_memory) > self.replay_start_size and self.i_action % 4 == 0:
            sample = zip(*self.replay_memory.sample(self.minibatch_size))

            s0 = np.array(sample[0], dtype=theano.config.floatX).reshape(self.minibatch_size, 4, self.screen_size, self.screen_size)

            a0 = np.array(sample[1], dtype=np.int8).reshape(self.minibatch_size, self.n_actions)

            r0 = np.array(sample[2], dtype=np.int16).reshape(self.minibatch_size, 1)

            future_reward_indicators = np.array(sample[3], dtype=np.int8).reshape(self.minibatch_size, 1)

            s1 = np.array(sample[4], dtype=theano.config.floatX).reshape(self.minibatch_size, 4, self.screen_size, self.screen_size)

            t = self.train_fn(s0, a0, r0, s1, future_reward_indicators)

            self.n_parameter_updates += 1

            self.log('{i_frame} | loss_elems: '.format(i_frame=self.i_action), t[1])
            self.log('{i_frame} | y, q: '.format(i_frame=self.i_action), t[2], t[3])
            self.log('{i_frame} | out: '.format(i_frame=self.i_action), t[4])
            self.log('{i_frame} | out_stale: '.format(i_frame=self.i_action), t[5])

            if self.n_parameter_updates % self.target_network_update_frequency == 0:
                self._update_network_stale()

        if self.i_action % self.save_every_n_frames == 100:
            filename = 'weights/weights_' + str(self.i_action) + '.npz'
            print("{i_frame} | File saved: {filename}".format(i_frame=self.i_action, filename=filename))
            np.savez(filename, *lasagne.layers.get_all_param_values(self.network))

    def __str__(self):
        return """
        self.mood_q = {self.mood_q}
        self.last_q = {self.last_q}
        self.n_parameter_updates = {self.n_parameter_updates}
        self.replay_memory = {self.replay_memory}
        self.alpha = {self.alpha}
        self.save_every_n_frames = {self.save_every_n_frames}
        self.final_exploration_frame = {self.final_exploration_frame}
        self.replay_start_size = {self.replay_start_size}
        self.i_action = {self.i_action}
        self.state = {self.state}
        self.initial_epsilon = {self.initial_epsilon}
        self.final_epsilon = {self.final_epsilon}
        self.epsilon = {self.epsilon}
        self.gamma = {self.gamma}
        self.log_frequency = {self.log_frequency}
        self.minibatch_size = {self.minibatch_size}
        self.target_network_update_frequency = {self.target_network_update_frequency}
        """.format(**{'self': self})


def build_loss(out, out_stale, a0_var, r0_var, future_reward_indicator_var, gamma):
    # s0_var mini_batch x 4 x,80 x 80
    # a0_var mini_batch x 1,
    # r0_var mini_batch x 1,
    # s1_mini_batch x 4 x 80 x 80
    future_reward_indicator_var.tag.test_value = np.random.rand(32, 1).astype(dtype=np.int8)
    r0_var.tag.test_value = np.random.rand(32, 1).astype(dtype=np.int16)
    a0_var.tag.test_value = np.random.rand(32, 6).astype(dtype=np.int8)
    out.tag.test_value = np.random.rand(1, 6).astype(dtype=theano.config.floatX)
    out_stale.tag.test_value = np.random.rand(32, 6).astype(dtype=theano.config.floatX)

    y = r0_var + gamma * future_reward_indicator_var * T.max(out_stale, axis=1, keepdims=True)  # 32 x 1
    q = T.sum(a0_var * out, axis=1, keepdims=True)  # 32 x 1
    err = y - q
    theano.shared

    quadratic_part = T.minimum(abs(err), 1)
    linear_part = abs(err) - quadratic_part
    loss = 0.5 * quadratic_part ** 2 + linear_part
    return T.mean(loss), loss, y, q
