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
                 max_actions_per_game,
                 skip_n_frames_after_lol,
                 tester):
        self.game = game
        self.algo = algo
        self.game_visualizer = game_visualizer
        self.repeat_action = repeat_action
        self.phi = phi
        self.skip_n_frames_after_lol = skip_n_frames_after_lol
        self.max_actions_per_game = max_actions_per_game
        self.run_ave = 0
        self.tester = tester

    def teach(self, i_total_action, total_n_actions):
        game_rewards = []
        while i_total_action < total_n_actions:
            game_reward, i_total_action = self.single_episode(i_total_action, total_n_actions)
            game_rewards.append(game_reward)
        return game_rewards

    def single_episode(self, i_total_action, total_n_actions):

        if self.game.finished:
            self.game.reset_game()
            self.algo.init_state(self.phi(self.game.get_state()))

        i_action = 0
        while not self.game.finished and i_action < self.max_actions_per_game and i_total_action < total_n_actions:
            i_action += 1
            i_total_action += 1

            self.single_action()

        # if not self.game.finished:
        #     print "Failure."

        # print "Game reward: " + str(self.game.cum_reward)
        # print ""
        #
        # self.run_ave = self.run_ave * 0.999 + 0.001 * self.game.cum_reward
        # print "Running average: " + str(self.run_ave)

        return self.game.cum_reward, i_total_action

    def single_action(self):
        old_state = self.phi(self.game.get_state())
        action = self.algo.action(old_state)

        import numpy as np

        rewards, lols = zip(*[self.game.input(action) for _ in range(self.repeat_action)])
        rep_reward = np.sum(rewards)
        lol = np.any(lols)
        new_state = self.phi(self.game.get_state())

        if not self.tester:
            exp = Experience(old_state, action, rep_reward, new_state, lol)
            self.algo.feedback(exp)

        self.game_visualizer.show(new_state)

        if lol:
            for _ in range(self.skip_n_frames_after_lol):
                self.game.input(action)

        self.game_visualizer.show(new_state)


def teach_and_test(teacher, tester, n_epochs, algo_initial_state_file=None):
    import numpy as np
    epoch_size = 50000
    frames_to_test_on = 10000

    if algo_initial_state_file is None:
        start_epoch = 0
    else:
        import re
        start_epoch = int(re.match(r".*epoch_([0-9]*).npz", algo_initial_state_file).groups()[0]) + 1
        print("Continuing from epoch: {start_epoch}".format(start_epoch=start_epoch))
        teacher.algo.set_state(np.load(algo_initial_state_file)['arr_0'])
        teacher.algo.i_action = start_epoch * epoch_size

    for i_epoch in range(start_epoch, n_epochs):
        import time
        start = time.time()
        print(teacher.algo.epsilon)
        rewards = teacher.teach(i_epoch*epoch_size, (i_epoch+1) * epoch_size)
        end = time.time()
        print("Epoch {i_epoch} mean training result {result} on {n} games in {t} seconds.".format(i_epoch=i_epoch, result=np.mean(rewards), n=len(rewards), t=end-start))
        algo_state = teacher.algo.get_state()
        filename = 'weights/epoch_' + str(i_epoch) + '.npz'

        np.savez(filename, algo_state)
        print("Algo state saved: {filename}".format(filename=filename))


        tester.algo.set_state(algo_state)

        start = time.time()
        rewards = tester.teach(0, frames_to_test_on)
        end = time.time()
        print(tester.algo.epsilon)
        print("Epoch {i_epoch} mean result: {result} on {n} games in {t} seconds.".format(i_epoch=i_epoch, result=np.mean(rewards), n=len(rewards), t=end-start))


