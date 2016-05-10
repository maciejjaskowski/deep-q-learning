from __future__ import division

from collections import namedtuple
import random
import numpy as np

Point = namedtuple('Point', 'x y')
Experience = namedtuple('Experience', 's0 a0 r0 s1 lol')


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
    def __init__(self, game, algo, game_visualizer, repeat_action,
                 max_actions_per_game,
                 skip_n_frames_after_lol,
                 tester):
        self.game = game
        self.algo = algo
        self.game_visualizer = game_visualizer
        self.repeat_action = repeat_action
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

        i_action = 0
        while not self.game.finished and i_action < self.max_actions_per_game and i_total_action < total_n_actions:
            i_action += 1
            i_total_action += 1

            self.single_action()


        print("Game reward | " + str(self.game.cum_reward), i_total_action)

        return self.game.cum_reward, i_total_action

    def single_action(self):
        action = self.algo.action()

        rep_reward, lol = self.game.input(action, self.repeat_action)
        new_state = self.game.get_state()

        self.algo.feedback(action, rep_reward, lol, new_state)

        self.game_visualizer.show(new_state)

        if lol:
            for _ in range(self.skip_n_frames_after_lol):
                self.game.input(action)

        self.game_visualizer.show(new_state)


def teach_and_test(teacher, tester, n_epochs, frames_to_test_on, epoch_size, state_dir, algo_initial_state_file=None):
    import numpy as np
    import os, re

    def extract_epoch(f):
        return int(re.match(r".*epoch_([0-9]*).npz", f).groups()[0])

    if algo_initial_state_file is None:
        state_files = [(extract_epoch(f), os.path.join(state_dir,f)) for f in os.listdir(state_dir) if f.startswith("epoch_")]
        if len(state_files) > 0:
            algo_initial_state_file = max(state_files)[1]
            print("Latest file with state found: {state_file}".format(state_file=algo_initial_state_file))

    if algo_initial_state_file is None:
        start_epoch = 0
    else:
        print("Using file with state: {state_file}".format(state_file=algo_initial_state_file))
        import re
        start_epoch = extract_epoch(algo_initial_state_file) + 1
        teacher.algo.set_state(np.load(algo_initial_state_file)['arr_0'])
        teacher.algo.i_action = start_epoch * epoch_size
        print("Continuing from epoch: {start_epoch} and action: {i_action}".format(
            start_epoch=start_epoch, i_action=teacher.algo.i_action))

    for i_epoch in range(start_epoch, n_epochs):
        import time
        start = time.time()
        print("Epsilon | {epsilon}".format(epsilon=teacher.algo.epsilon))
        rewards = teacher.teach(i_epoch*epoch_size, (i_epoch+1) * epoch_size)
        end = time.time()
        teacher.game.stop_game()
        print("Epoch {i_epoch} mean training result {result} on {n} games and {epoch_size} actions in {t} seconds.".format(
            i_epoch=i_epoch, result=np.mean(rewards), n=len(rewards), t=end-start, epoch_size=epoch_size))

        algo_state = teacher.algo.get_state()
        filename = '{state_dir}/epoch_{i_epoch}.npz'.format(state_dir=state_dir, i_epoch=i_epoch)
        np.savez(filename, algo_state)
        print("Algo state saved: {filename}".format(filename=filename))
        tester.algo.set_state(algo_state)

        start = time.time()
        rewards = tester.teach(0, frames_to_test_on)
        end = time.time()
        tester.game.stop_game()
        print(tester.algo.epsilon)
        print("Epoch {i_epoch} mean validation result: {result} on {n} games in {t} seconds.".format(
            i_epoch=i_epoch, result=np.mean(rewards), n=len(rewards), t=end-start))


