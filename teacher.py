from __future__ import division

from collections import namedtuple
import random
import numpy as np


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
    def __init__(self, game, algo, game_visualizer, phi, repeat_action,
                 i_total_action,
                 total_n_actions,
                 max_actions_per_game,
                 skip_n_frames_after_lol,
                 run_test_every_n):
        self.game = game
        self.algo = algo
        self.game_visualizer = game_visualizer
        self.repeat_action = repeat_action
        self.phi = phi
        self.skip_n_frames_after_lol = skip_n_frames_after_lol
        self.total_n_actions = total_n_actions
        self.i_total_action = i_total_action
        self.max_actions_per_game = max_actions_per_game
        self.run_ave = 0
        self.run_test_every_n = run_test_every_n

    def teach(self):
        while self.i_total_action < self.total_n_actions:
            self.single_episode(feedback=True, after_action_callback=self.run_test_every)

    def run_test_every(self):
        if self.i_total_action % self.run_test_every_n == 0:
            self.algo.test_mode = True
            res = [self.single_episode(feedback=False, after_action_callback=self.do_nothing) for _ in range(400)]
            print(res)
            print("Test | {i_total_action} | {mean_reward}".format(
                i_total_action=self.i_total_action,
                mean_reward=np.mean(res)))

            self.algo.test_mode = False

    def do_nothing(self):
        pass

    def single_episode(self, feedback, after_action_callback):
        self.game.reset_game()
        self.algo.init_state(self.phi(self.game.get_state()))

        i_action = 0

        while not self.game.finished and i_action < self.max_actions_per_game:
            if feedback:
                i_action += 1
                self.i_total_action += 1

            self.single_action(feedback=feedback)
            after_action_callback()

        if not self.game.finished:
            print "Failure."

        # if feedback:
        #     print "Game reward: " + str(self.game.cum_reward)
        #     print ""
        #
        #     self.run_ave = self.run_ave * 0.999 + 0.001 * self.game.cum_reward
        #     print "Running average: " + str(self.run_ave)

        self.game_visualizer.next_game()
        return self.game.cum_reward

    def single_action(self, feedback):
        old_state = self.phi(self.game.get_state())
        action = self.algo.action(old_state)

        import numpy as np

        rewards, lols = zip(*[self.game.input(action) for _ in range(self.repeat_action)])
        rep_reward = np.sum(rewards)
        lol = np.any(lols)
        new_state = self.phi(self.game.get_state())

        exp = Experience(old_state, action, rep_reward, new_state, lol)
        if feedback:
            self.algo.feedback(exp)

        self.game_visualizer.show(new_state)

        if lol:
            for _ in range(self.skip_n_frames_after_lol):
                self.game.input(action)

        self.game_visualizer.show(new_state)
