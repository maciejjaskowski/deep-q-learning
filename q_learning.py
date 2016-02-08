from __future__ import division

from collections import namedtuple
from copy import deepcopy
from random import random, choice, uniform, sample, randrange
from time import sleep

from bisect import bisect
import itertools
import numpy as np
import pandas as pd

from blist import sortedlist


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


class Tilings: # n_tilings = 5, n = 9

  def __init__(self, x, y, n, n_tilings):
    self.dx = (x[1] - x[0])
    self.dy = (y[1] - y[0])
    self.range_x = np.array(range(n-1)) / (n-1) * self.dx + x[0] 
    self.range_y = np.array(range(n-1)) / (n-1) * self.dy + y[0] 
    self.n = n
    self.n_tilings = n_tilings

  def tiling(self):
    distortion_x = [ x + random() * self.dx / (self.n-1) for x in self.range_x]
    distortion_y = [ y + random() * self.dy / (self.n-1) for y in self.range_y]
    return distortion_x, distortion_y

  def tilings(self): 
    return [ self.tiling() for i in range(self.n_tilings)]

  def one_at(self, i):
    result = np.zeros(self.n * self.n)
    result[i] = 1
    return result

  def calc(self):
    ts = self.tilings()
    return (lambda p: tuple([(bisect(tiling[0], p[0]), bisect(tiling[1], p[1])) for tiling in ts]), ts)
   
class RandomAlgo:
    def __init__(self, legal_actions):
      self.legal_actions = legal_actions

    def action(self):
      while True:
        yield self.legal_actions[randrange(len(self.legal_actions))]

    def feedback(self, x):
        pass    


class Teacher:
    def __init__(self, new_game, algo, game_visualizer, repeat_action = 1):
      self.new_game = new_game
      self.algo = algo
      self.game_visualizer = game_visualizer
      self.algo_input = self.algo.action()
      self.repeat_action = repeat_action


    def teach(self, episodes):
      return [self.single_play(15000) for i in range(episodes)]

    def single_play(self, n_steps = float("inf")):
      Game = self.new_game()

      i_steps = 0

      while not Game.finished and i_steps < n_steps:
        i_steps += 1
        exp = self.single_step(Game)

      if Game.finished:
        print "Finished after ", i_steps, " steps"    
      else:
        print "Failure."  

      print Game.cum_reward  

      self.game_visualizer.next_game()

      return (i_steps, Game.cum_reward)

    def single_step(self, Game):

      old_state = Game.get_state()
      old_cum_reward = Game.cum_reward

      action = next(self.algo_input)
      for i in range(self.repeat_action):
        Game.input(action)

      exp = Experience(old_state, action, Game.cum_reward - old_cum_reward, Game.get_state(), Game.finished)
      self.algo.feedback(exp)
      
      self.game_visualizer.show(Game)  
      return exp
