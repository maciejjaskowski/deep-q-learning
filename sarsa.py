from __future__ import division
import numpy as np
from random import random, randint


class SARSALambdaGradientDescent:
    def __init__(self, n_actions, theta_len, state_adapter=lambda x: x):
        self.lmbda = 0.8
        self.gamma = 0.7
        self.alpha = 0.1
        self.epsilon = 0.1
        self.phi = state_adapter
        self.log_freq = 0.05

        self.n_actions = n_actions
        self.state = None

        self.theta = np.random.rand(self.n_actions, theta_len) / 80 / 80 / 4

        self.e = np.zeros([self.n_actions, theta_len])
        self.next_action = None

    def init_state(self, state):
        self.state = self.phi(state)
        self.next_action = self._action(self.state)

    def _q(self, state, action):
        return sum(self.theta[action][state])

    def action(self):
        return self.next_action

    def _action(self, state):
        if random() < self.epsilon:
            return randint(0, self.n_actions - 1)
        else:
            return self._best_action(state)

    def _best_action(self, state):
        return max([(action, self._q(state, action)) for action in xrange(self.n_actions)],
                   key=lambda opt: opt[1])[0]

    def _pi_value(self, state):
        return max([(action, self._q(state, action)) for action in xrange(self.n_actions)],
                   key=lambda opt: opt[1])[1]

    def feedback(self, exp):
        a0 = exp.a0
        s0 = self.phi(exp.s0)
        s1 = self.phi(exp.s1)
        a1 = self._action(s1)
        r0 = exp.r0
        game_over = int(exp.game_over)

        for a in xrange(self.n_actions):
            self.e[a][s1] = 0
        self.e[a1][s1] = 1.0

        delta = r0 + (1 - exp.game_over) * (self.gamma * self._q(s1, a1) - self._q(s0, a0))
        if random() < self.log_freq:
            print ("game_over ", game_over, "delta ", delta)
            print ("r", r0, "g", self.gamma, "q1", self._q(s1, a1), "q0", self._q(s0, a0))
            print ("a0", "a1", a0, a1)
            print ("s1", np.shape(s1))

        self.theta += (self.alpha * delta) * self.e
        self.e *= self.gamma
        self.e *= self.lmbda

        self.state = s1
        self.next_action = a1
