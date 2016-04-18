from __future__ import division

from collections import namedtuple
import random


Point = namedtuple('Point', 'x y')
Experience = namedtuple('Experience', 's0 a0 r0 s1 game_over')


class GameNoVisualizer:
    def show(self, game):
        pass

    def next_game(self):
        pass


class RandomAlgo:
    def __init__(self, n_actions):
        self.n_actions = n_actions

    def init_state(self, state):
        pass

    def action(self):
        return random.randint(0, self.n_actions - 1)

    def feedback(self, x):
        pass


class ConstAlgo:
    def __init__(self, const_actions):
        self.const_actions = const_actions
        self.i = 0

    def init_state(self, state):
        pass

    def action(self):
        self.i += 1
        return self.const_actions[self.i % len(self.const_actions)]

    def feedback(self, x):
        pass


class Teacher:
    def __init__(self, new_game, algo, game_visualizer, phi, repeat_action,
                 i_total_action,
                 total_n_actions,
                 max_actions_per_game,
                 skip_n_frames_after_lol):
        self.new_game = new_game
        self.algo = algo
        self.game_visualizer = game_visualizer
        self.repeat_action = repeat_action
        self.phi = phi
        self.skip_n_frames_after_lol = skip_n_frames_after_lol
        self.total_n_actions = total_n_actions
        self.i_total_action = i_total_action
        self.max_actions_per_game = max_actions_per_game

    def teach(self):
        while self.i_total_action < self.total_n_actions:
            self.single_episode()

    def single_episode(self):
        game = self.new_game()
        self.algo.init_state(self.phi(game.get_state()))

        i_action = 0
        while not game.finished and i_action < self.max_actions_per_game:
            i_action += 1
            self.single_action(game)

        if game.finished:
            print "Finished after ", i_action, " actions"
        else:
            print "Failure."

        print "Game reward: " + str(game.cum_reward)
        print ""

        self.game_visualizer.next_game()

    def single_action(self, game):

        action = self.algo.action()

        import numpy as np
        old_state = self.phi(game.get_state())
        rewards, lols = zip(*[game.input(action) for _ in range(self.repeat_action)])
        rep_reward = np.sum(rewards)
        lol = np.any(lols)
        new_state = self.phi(game.get_state())

        exp = Experience(old_state, action, rep_reward, new_state, lol)
        self.algo.feedback(exp)

        self.game_visualizer.show(new_state)

        if lol:
            for _ in range(self.skip_n_frames_after_lol):
                game.input(action)

        self.game_visualizer.show(new_state)
