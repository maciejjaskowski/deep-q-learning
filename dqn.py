# Alt + Shift + E

from __future__ import print_function
from __future__ import division

import numpy as np
import theano
import theano.tensor as T

import lasagne


class ReplayMemory(object):
    def __init__(self, size=1000000, grace=10000, tuple_len=4):
        self.max_size = size
        self.grace = grace
        self.list = []
        self.tuple_len = tuple_len

    def init_state(self, s0):
        if len(self.list) > 0:
            self.list[-1] = s0
        else:
            self.list.append(s0)

    def append(self, a0, r0, fri, s1):
        el = [a0, r0, fri, s1]

        self.list.extend(el)
        if len(self) > self.max_size + self.grace * self.tuple_len:
            self.list = self.list[self.grace * self.tuple_len:]

    def sample(self, sample_size):
        import random
        indices = random.sample(xrange(len(self)), sample_size)
        return [self[i] for i in indices]

    def __len__(self):
        return int((len(self.list) - 1) / self.tuple_len)

    def __getitem__(self, idx):
        return tuple(self.list[idx * self.tuple_len:idx * self.tuple_len + self.tuple_len + 1])



class Stacker(object):
    def __init__(self, n_stack, shape, max_memory_size, grace):
        self.n_stack = n_stack
        self.list = []
        self.shape = shape
        self.max_memory_size = max_memory_size
        self.grace = grace

    def append(self, x):
        # print('append', x)
        if len(self.list) > self.max_memory_size + self.grace:
            self.list = self.list[self.grace:]
        self.list.append(x)

    def __len__(self):
        return max(0,len(self.list) - self.n_stack + 1)

    def __getitem__(self, item):
        l = len(self.list) - 1
        import collections
        if isinstance(item, slice):
            item = slice(l - item.stop - self.n_stack + 1, l - item.start + 1, item.step)
            ret = [self.list[item]]
        elif isinstance(item, collections.Sequence):
            ret = [self.list[l - i - self.n_stack + 1:l - i + 1] for i in item]
        else:
            if len(self) <= item:
                raise RuntimeError("" + str(item))
            if item < 0:
                item = l + item + 1
            ret = self.list[l - item - self.n_stack + 1:l - item + 1]
        return np.reshape(np.stack(ret, axis=0), (-1, self.n_stack) + self.shape)


class Memory(object):
    def __init__(self, max_size, n_channels, screen_size,
                 game_variables, n_actions, input_n_last_actions,
                 size_grace):
        self.s_stacker = Stacker(n_channels, screen_size, max_size, size_grace)
        self.a_stacker = Stacker(1, (), max_size, size_grace)
        self.r_stacker = Stacker(1, (), max_size, size_grace)
        self.fri_stacker = Stacker(1, (), max_size, size_grace)
        self.input_n_last_actions = input_n_last_actions
        self.n_actions = n_actions

        if self.input_n_last_actions > 0:
            self.action_stacker = Stacker(self.input_n_last_actions, (self.n_actions,), max_size, size_grace)

        self.vars_stacker = [Stacker(n_stack, (), max_size, size_grace) for n_stack in game_variables]
        self.index_grace = 30
        self.game_variables = game_variables
        self.n_actions = n_actions
        self.input_n_last_actions = input_n_last_actions

    def __len__(self):
        return len(self.s_stacker)

    def sample(self, size):
        import random
        indices = random.sample(xrange(0, len(self.s_stacker) - self.index_grace), size)
        indices_m1 = [i+1 for i in indices]

        #print([self.vars_stacker[i][indices_m1].reshape(-1, self.game_variables[i]).shape for i in xrange(len(s1[1]))])
        inputs0 = [self.vars_stacker[i][indices_m1].reshape(-1, self.game_variables[i])
                   for i in xrange(len(self.vars_stacker))]
        inputs1 = [self.vars_stacker[i][indices]   .reshape(-1, self.game_variables[i])
                   for i in xrange(len(self.vars_stacker))]
        if self.input_n_last_actions > 0:
            inputs0.append(self.action_stacker[indices_m1].reshape(-1, self.input_n_last_actions * self.n_actions))
            inputs1.append(self.action_stacker[indices]   .reshape(-1, self.input_n_last_actions * self.n_actions))

        if len(inputs0) > 0:
            add0 = np.concatenate(inputs0, axis=1)
            add1 = np.concatenate(inputs1, axis=1)
        else:
            add0 = np.array([], dtype=theano.config.floatX).reshape(size, 0)
            add1 = np.array([], dtype=theano.config.floatX).reshape(size, 0)

        return {'s0': self.s_stacker[indices_m1],
                'add0': add0,
                'a0': self.a_stacker[indices],
                'r0': self.r_stacker[indices],
                'fri': self.fri_stacker[indices],
                's1': self.s_stacker[indices],
                'add1': add1}

    def append(self, a0, r0, fri, s1, add1):
        self.s_stacker.append(s1)

        for i in xrange(len(self.vars_stacker)):
            self.vars_stacker[i].append(add1[i])
        self.a_stacker.append(a0)
        self.r_stacker.append(r0)
        self.fri_stacker.append(fri)
        if self.input_n_last_actions > 0:
            self.action_stacker.append(a0)

    def last(self):
        if self.input_n_last_actions > 0:
            last_actions = [self.action_stacker[0]]
            # print(last_actions)
        else:
            last_actions = []
        input = [self.vars_stacker[i][0].reshape(1,-1) for i in xrange(len(self.vars_stacker))] + [np.reshape(last_actions, (1,-1))]
        # print(input)
        return (self.s_stacker[0], np.concatenate(input, axis=1).astype(dtype=theano.config.floatX))



class Agent(object):
    def __init__(self, gamma, n_actions, n_channels, screen_size, max_memory_size, memory_grace, no_learning, dqn,
                 game_variables, input_n_last_actions):
        self.screen_width, self.screen_height = screen_size
        self.no_learning = no_learning
        self.a_lookup = np.eye(n_actions, dtype=np.int8)
        self.dqn = dqn
        self.n_actions = n_actions
        self.n_parameter_updates = 0
        self.save_every_n_frames = 100000
        self.n_channels = n_channels

        self.final_exploration_frame = 1000000
        self.replay_start_size = 50000
        self.i_action = 0

        self.clip_rewards = True

        self.initial_epsilon = 1
        self.final_epsilon = 0.1
        self.epsilon = self.initial_epsilon
        self.gamma = gamma
        self.game_variables = game_variables
        self.input_n_last_actions = input_n_last_actions

        self.memory = Memory(max_size=max_memory_size, size_grace=memory_grace,
                             input_n_last_actions=self.input_n_last_actions,
                             game_variables=game_variables, n_actions=n_actions,
                             screen_size=screen_size,
                             n_channels=n_channels)

        self.log_frequency = 1

        self.minibatch_size = 32
        # self.replay_memory_size = 1000000

        self.target_network_update_frequency = 10000

    def init_state(self, state):
        pass
        # self.s_stacker.append(state[0])
        # self.vars_stacker[0].append(state[1])

    def _action(self, _state):
        import random
        if self.i_action < self.final_exploration_frame:
            self.epsilon = (self.final_epsilon - self.initial_epsilon) * (
                self.i_action / self.final_exploration_frame) + self.initial_epsilon
        else:
            self.epsilon = self.final_epsilon

        if random.random() < self.epsilon:
            random_action = random.randint(0, self.n_actions - 1)
            # print("{i_frame} | a | 1 | {action} | {q}".format(i_frame=self.i_action, action=random_action, q=q))
            return random_action
        else:
            # print("state", _state)
            q = self.dqn.forward(*_state)

            best_action = np.argmax(q)
            # print("{i_frame} | a | 0 | {action} | {q}".format(i_frame=self.i_action, action=best_action, q=q))
            return best_action

    def action(self):
        if len(self.memory) < 30:
            import random
            return random.randint(0, self.n_actions-1)
        return self._action(self.memory.last())

    def feedback(self, a0, r0, lol, s1):
        # exp -> s0 a0 r0 s1 game_over
        self.i_action += 1

        if self.clip_rewards:
            r0 = min(1, max(-1, r0))

        fri = 1 - int(lol)

        self.memory.append(a0=self.a_lookup[a0], r0=r0, fri=fri, s1=s1[0], add1=s1[1])

        if not self.no_learning and len(self.memory) - 30 > self.replay_start_size and self.i_action % 4 == 0:
            self.dqn.train(**self.memory.sample(self.minibatch_size))

        self.n_parameter_updates += 1
        if self.n_parameter_updates % self.target_network_update_frequency == 0:
            self.dqn._update_network_stale()

        return

    def get_state(self):
        return self.dqn.get_state()

    def set_state(self, param_values):
        self.dqn.set_state(param_values)


class DQN(object):
    def __init__(self, gamma, screen_size, n_actions, build_network, n_additional_params, n_channels, updates):
        self.screen_width, self.screen_height = screen_size
        self.gamma = gamma
        self.i_train = 0
        self.n_channels = n_channels
        self.n_additional_params = n_additional_params
        # if self.n_additional_params <= 0:
        #     raise RuntimeError("n_additional_params must be > 0.")

        s0_var = T.tensor4("s0", dtype=theano.config.floatX)
        a0_var = T.bmatrix("a0")
        r0_var = T.wcol("r0")
        s1_var = T.tensor4("s1", dtype=theano.config.floatX)
        future_reward_indicator_var = T.bcol("future_reward_indicator")

        self.n_actions = n_actions

        additional_input_var0 = T.matrix("add0", dtype=theano.config.floatX)
        additional_input_var1 = T.matrix("add1", dtype=theano.config.floatX)

        self.network = build_network(n_actions=self.n_actions, input_var=T.cast(s0_var, 'float32') / np.float32(256),
                                     n_stack_frames=self.n_channels,
                                     screen_size=(self.screen_height, self.screen_width),
                                     n_additional_params=n_additional_params,
                                     additional_input_var=additional_input_var0)
        print("Compiling forward.")


        self.forward = theano.function([s0_var, additional_input_var0],
                                       lasagne.layers.get_output(self.network, deterministic=True),
                                       on_unused_input='warn')

        self.network_stale = build_network(n_actions=self.n_actions, input_var=T.cast(s1_var, 'float32') / np.float32(256),
                                           n_stack_frames=self.n_channels,
                                           screen_size=(self.screen_height, self.screen_width),
                                           n_additional_params=n_additional_params,
                                           additional_input_var=additional_input_var1)
        print("Compiling forward_stale.")
        self.forward_stale = theano.function([s1_var, additional_input_var1],
                                             lasagne.layers.get_output(self.network_stale, deterministic=True),
                                             on_unused_input='warn')

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

        out1 = lasagne.layers.get_output(lasagne.layers.get_all_layers(self.network)[1])

        print("Compiling train_fn.")
        self.train_fn = theano.function([s0_var, additional_input_var0, a0_var, r0_var, s1_var, additional_input_var1, future_reward_indicator_var],
                                        [self.loss, self.err, T.transpose(__y), T.transpose(__q), out, out_stale, out1],
                                        updates=updates(self.loss, params),
                                        on_unused_input='warn')
        print("Compiling loss_fn.")
        self.loss_fn = theano.function([s0_var, additional_input_var0, a0_var, r0_var, s1_var, additional_input_var1,
                                        future_reward_indicator_var], self.loss, on_unused_input='warn')

    def train(self, s0, add0, a0, r0, fri, s1, add1):

        s0 = np.array(s0, dtype=theano.config.floatX).reshape(-1, self.n_channels, self.screen_width, self.screen_height)
        a0 = np.array(a0, dtype=np.int8).reshape(-1, self.n_actions)
        r0 = np.array(r0, dtype=np.int16).reshape(-1, 1)
        fri = np.array(fri, dtype=np.int8).reshape(-1, 1)
        s1 = np.array(s1, dtype=theano.config.floatX).reshape(-1, self.n_channels, self.screen_width, self.screen_height)

        additional0 = np.array(add0, dtype=theano.config.floatX)
        additional1 = np.array(add1, dtype=theano.config.floatX)

        t = self.train_fn(s0, additional0, a0, r0, s1, additional1, fri)

        self.i_train += 1
        if self.i_train % 800 == 0:
            print('{i_frame} | loss_elems: '.format(i_frame=self.i_train), t[1])
            print('{i_frame} | y, q: '.format(i_frame=self.i_train), t[2], t[3])
            print('{i_frame} | out: '.format(i_frame=self.i_train), t[4])
            print('{i_frame} | out_stale: '.format(i_frame=self.i_train), t[5])

    def _update_network_stale(self):
        print("update stale")
        lasagne.layers.set_all_param_values(self.network_stale, lasagne.layers.get_all_param_values(self.network))

    def get_state(self):
        return lasagne.layers.get_all_param_values(self.network)

    def set_state(self, param_values):
        lasagne.layers.set_all_param_values(self.network, param_values)
        lasagne.layers.set_all_param_values(self.network_stale, param_values)


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
    return T.sum(loss), loss, y, q
