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
    def __init__(self, new_game, algo, game_visualizer, phi, repeat_action=1, sleep_seconds=0):
        self.new_game = new_game
        self.algo = algo
        self.game_visualizer = game_visualizer
        self.repeat_action = repeat_action
        self.phi = phi
        self.old_state = None
        self.sleep_seconds = sleep_seconds

    def teach(self, episodes):
        return [self.single_episode(15000) for i in range(episodes)]

    def single_episode(self, n_steps=float("inf")):
        game = self.new_game()
        self.old_state = self.phi(game.get_state())
        self.algo.init_state(self.old_state)

        i_steps = 0

        while not game.finished and i_steps < n_steps:
            i_steps += 1
            exp, elapsed_time = self.single_step(game)
            if i_steps % 10000 < 10:
                print("elapsed time: {elapsed_time}".format(elapsed_time=elapsed_time))

        if game.finished:
            print "Finished after ", i_steps, " steps"
        else:
            print "Failure."

        print "Game reward: " + str(game.cum_reward)
        print ""

        self.game_visualizer.next_game()

        return i_steps, game.cum_reward

    def single_step(self, game):
        import time
        time_start = time.time()
        old_cum_reward = game.cum_reward

        action = self.algo.action()
        #print("action: ", action)

        new_state = None
        skip = 0
        for i in range(self.repeat_action):
            skip = max(skip, game.input(action))
            new_state = self.phi(game.get_state())

        exp = Experience(self.old_state, action, game.cum_reward - old_cum_reward, new_state, game.finished)
        self.algo.feedback(exp)

        self.game_visualizer.show(new_state)

        for i in range(skip):
            game.input(action)
            new_state = self.phi(game.get_state())
            self.game_visualizer.show(new_state)

        self.old_state = new_state
        if self.sleep_seconds != 0:
            import time
            time.sleep(self.sleep_seconds)

        time_end = time.time()

        return exp, time_end - time_start
