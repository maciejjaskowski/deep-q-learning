from __future__ import division

from collections import namedtuple
from random import randrange
from copy import deepcopy


Point = namedtuple('Point', 'x y')
Experience = namedtuple('Experience', 's0 a0 r0 s1 game_over')


class GameNoVisualizer:
    def show(self, game):
        pass

    def next_game(self):
        pass


class EveryNVisualizer:
    def __init__(self, n, visualizer):
        self.n = n
        self.right_visualizer = visualizer
        self.visualizer = GameNoVisualizer()
        self.i = 0

    def show(self, game):
        self.visualizer.show(game)

    def next_game(self):
        self.i += 1
        if (self.i % self.n == self.n - 1):
            self.visualizer = self.right_visualizer
        else:
            self.visualizer = GameNoVisualizer()


class RandomAlgo:
    def __init__(self, legal_actions):
        self.legal_actions = legal_actions

    def action(self):
        while True:
            yield self.legal_actions[randrange(len(self.legal_actions))]

    def feedback(self, x):
        pass


class Teacher:
    def __init__(self, new_game, algo, game_visualizer, phi, repeat_action=1):
        self.new_game = new_game
        self.algo = algo
        self.game_visualizer = game_visualizer
        self.algo_input = self.algo.action()
        self.repeat_action = repeat_action
        self.phi = phi

    def teach(self, episodes):
        return [self.single_play(15000) for i in range(episodes)]

    def single_play(self, n_steps=float("inf")):
        game = self.new_game()

        i_steps = 0

        while not game.finished and i_steps < n_steps:
            i_steps += 1
            exp = self.single_step(game)

        if game.finished:
            print "Finished after ", i_steps, " steps"
        else:
            print "Failure."

        print game.cum_reward

        self.game_visualizer.next_game()

        return i_steps, game.cum_reward

    def single_step(self, game):

        old_state = self.phi(game.get_state())
        old_cum_reward = game.cum_reward

        action = next(self.algo_input)

        new_state = None
        for i in range(self.repeat_action):
            game.input(action)
            new_state = self.phi(game.get_state())

        exp = Experience(old_state, action, game.cum_reward - old_cum_reward, new_state, game.finished)
        self.algo.feedback(exp)

        self.game_visualizer.show(new_state)
        return exp
