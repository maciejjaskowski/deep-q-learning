from bisect import bisect
import itertools
import numpy as np
import pandas as pd
from random import random


class Tilings:  # n_tilings = 5, n = 9

    def __init__(self, x, y, n, n_tilings):
        self.dx = (x[1] - x[0])
        self.dy = (y[1] - y[0])
        self.range_x = np.array(range(n - 1)) / (n - 1) * self.dx + x[0]
        self.range_y = np.array(range(n - 1)) / (n - 1) * self.dy + y[0]
        self.n = n
        self.n_tilings = n_tilings

    def tiling(self):
        distortion_x = [x + random() * self.dx / (self.n - 1) for x in self.range_x]
        distortion_y = [y + random() * self.dy / (self.n - 1) for y in self.range_y]
        return distortion_x, distortion_y

    def tilings(self):
        return [self.tiling() for i in range(self.n_tilings)]

    def one_at(self, i):
        result = np.zeros(self.n * self.n)
        result[i] = 1
        return result

    def calc(self):
        ts = self.tilings()
        return (lambda p: tuple([(bisect(tiling[0], p[0]), bisect(tiling[1], p[1])) for tiling in ts]), ts)


class MountainCarGame:
    def __init__(self):
        self.position = -0.5  # random() * ( 0.5 - (-1.2)) + (-1.2)
        self.throttle = 0
        self.velocity = 0  # random() * (0.07 - (-0.07)) + (-0.07)
        self.cum_reward = 0
        self.finished = False

    def input(self, ch):
        import math
        if ch in self.get_actions():
            self.throttle = ch

        self.velocity = self.velocity + 0.001 * self.throttle - 0.0025 * math.cos(3 * self.position)
        if self.velocity < -0.07:
            self.velocity = -0.07
        if self.velocity > 0.07:
            self.velocity = 0.07

        self.position += self.velocity
        if self.position > 0.5:
            self.finished = True
        if self.position < -1.2:
            self.position = -1.2
            self.velocity = 0

        self.cum_reward = self.cum_reward - 1 + abs(self.position + 0.5)
        return self

    def get_actions(self):
        return [-1, 0, 1]

    def get_state(self):
        return self.position, self.velocity


def mountain_car_game_tilings_state_adapter(tile_in_row, n_tilings):
    my_tilings, til = Tilings((-1.2, 0.5), (-0.07, 0.07), tile_in_row, n_tilings).calc()
    return lambda (a, b): tuple([x + y * tile_in_row + i * tile_in_row * tile_in_row
                                 for (x, y), i in zip(my_tilings((a, b)), range(n_tilings * n_tilings * tile_in_row))])


class MountainCarGameVisualizer:
    def __init__(self, algo, print_every_n=10, state_adapter=lambda x: x):
        self.state_adapter = state_adapter
        self.algo = algo
        self.print_every_n = print_every_n
        self.i = -1

        import matplotlib.pyplot as plt
        self.fig = plt.figure()
        self.velocity_position = self.fig.add_subplot(1, 1, 1)
        # self.expected_reward = self.fig.add_subplot(2, 2, 2, projection = '3d')
        # self.direction = self.fig.add_subplot(2, 2, 3)

        self.hl, = self.velocity_position.plot([], [], color='k', linestyle='-')
        self.mi, = self.velocity_position.plot([], [], color='red', linestyle='-')
        self.ma, = self.velocity_position.plot([], [], color='red', linestyle='-')
        self.dir = [0, 0, 0]
        self.dir[0], = self.velocity_position.plot([], [], color='red', linestyle='', marker='<', ms=5)
        self.dir[1], = self.velocity_position.plot([], [], color='blue', linestyle='', marker='.', ms=1)
        self.dir[2], = self.velocity_position.plot([], [], color='green', linestyle='', marker='>', ms=5)
        self.velocity_position.set_xlim([-1.3, 0.6])
        self.velocity_position.set_ylim([-0.1, 0.1])

        self.history_x = []
        self.history_y = []

        self.xlim = [0, 0]

    def show(self, game):
        self.i += 1
        import matplotlib.pyplot as plt
        import matplotlib

        if len(self.history_x) > 100:
            self.history_x = self.history_x[1:]
            self.history_y = self.history_y[1:]

        newx, newy = self.state_adapter(game.get_state())
        self.history_x.append(newx)
        self.history_y.append(newy)
        self.xlim[0] = min(self.xlim[0], newx)
        self.xlim[1] = max(self.xlim[1], newx)

        if self.i % self.print_every_n == 0:
            self.hl.set_xdata(self.history_x)
            self.hl.set_ydata(self.history_y)
            self.mi.set_xdata([self.xlim[0]] * 2)
            self.mi.set_ydata([-0.08, 0.08])
            self.ma.set_xdata([self.xlim[1]] * 2)
            self.ma.set_ydata([-0.08, 0.08])

            pos = np.arange(-1.2, 0.5, 0.05)
            vel = np.arange(-0.07, 0.07, 0.005)

            direction = pd.DataFrame(
                [(_pos, _vel, self.algo._best_action(self.state_adapter((_pos, _vel)))) for _vel in vel for _pos in pos])
            direction.columns = ["pos", "vel", "throttle"]

            col = {-1: 'red', 0: 'blue', 1: 'green'}
            for name, group in direction.groupby('throttle'):
                # if (name >= 0):
                self.dir[name + 1].set_xdata(group['pos'])
                self.dir[name + 1].set_ydata(group['vel'])
            # self.expected_reward.plot_surface(Pos, Vel, expected_reward)

            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

            import time
            time.sleep(0.01)

            # print(game.get_state(), game.throttle)

    def next_game(self):
        pass
