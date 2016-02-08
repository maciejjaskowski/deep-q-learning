import sys
from random import randrange
from ale_python_interface import ALEInterface
from itertools import groupby
import numpy as np
import pygame



# Set USE_SDL to true to display the screen. ALE must be compilied
# with SDL enabled for this to work. On OSX, pygame init is used to
# proxy-call SDL_main.

#USE_SDL = False
#if USE_SDL:
#  if sys.platform == 'darwin':   
#    import pygame
#    pygame.init() 
#    ale.setBool('sound', False) # Sound doesn't work on OSX    
#  elif sys.platform.startswith('linux'):
#    ale.setBool('sound', True)
  

# ale.setString("record_screen_dir", "record")

# Load the ROM file
#ale.setBool('display_screen', True)  

ARR = {0: (0,0,0),
           6: (200, 200, 0),
           20: (0, 200, 200),
           52: (200, 0, 200),
           196: (196, 0, 0),
           226: (0, 226, 0), 
           246: (146, 0, 0)}
COLORS = sorted(ARR.keys())

mergeArr = {0: 0,
            6: 6,
            20: 20, #robaki
            52: 52, # oslony
            196: 196,
            226: 0,
            246: 0}

mergeArrValuesSet = set(mergeArr.values())
#mergeArrValuesSet.remove(0)
mergeArrValues = sorted(list(mergeArrValuesSet))

def init():

  pygame.init()
  rom_path = '/Users/maciej/Development/atari-roms'
  ale = ALEInterface()
  ale.setInt('random_seed', 123)
  ale.setBool('frame_skip', 1)
  ale.loadROM(rom_path + '/space_invaders.bin')
  ale.setFloat("repeat_action_probability", 0)
  return ale

def vectorize_single_group(vec):
    return map(lambda e: e in vec, mergeArrValues)

def vectorized(scr, desired_width, desired_height):
    grouped = \
      np.reshape(np.swapaxes(np.reshape(scr, (desired_width, 210 / desired_width, desired_height, 160 / desired_height)), 1,2), (desired_width, desired_height, 160 * 210 / desired_width / desired_height))
    return np.apply_along_axis(vectorize_single_group, axis = 2, arr = grouped)  

class SpaceInvadersGameVectorizedVisualizer:
  def __init__(self):

    w_orig = 160
    h_orig = 210

    

    self.desired_width = 14
    self.desired_height = 20
    self.screen = pygame.display.set_mode((self.desired_height * 16, self.desired_width * 16))\


  def show_vectorized(self, vec):
    rect = pygame.Surface((2, 14))
    border = pygame.Surface((16, 16))
    
    border.fill((255, 255, 255))
    for y in range(0, self.desired_width):
      for x in range(0, self.desired_height):
        #border_rect = pygame.Rect(x, y, 16, 16)    
        #self.screen.blit(border, (x*16, y*16))  
       
        for i_color in range(len(mergeArrValues)):
          if (vec[y][x][i_color]):
            rect.fill(ARR[COLORS[i_color]])
          else:
            rect.fill((0, 0, 0))
          self.screen.blit(rect, (x * 16 + 1 + i_color*2, y * 16 + 1))
        
    pygame.display.flip()   

  def show(self, game):
    self.show_vectorized(vectorized(game.get_state(), self.desired_width, self.desired_height))

  def next_game(self):
    pass  

class SpaceInvadersGame:

  def __init__(self, ale):
    self.ale = ale
    self.finished = False    
    self.cum_reward = 0
    self.state = ale.getScreen()

  def get_actions(self):
    return self.ale.getMinimalActionSet()  

  def input(self, action):
    #print ("action: ", action)
    self.cum_reward += self.ale.act(action)
    if (self.ale.game_over()):
      print ("finished!")
      self.finished = True
      self.ale.reset_game()

    self.state = self.ale.getScreen()
    return self

  def get_state(self):
    return self.state




    #def show(sqr):  
    #  square=pygame.Surface(1, 1)
    #  for y in range(0, 210, reduceby):
    #    for x in range(0, 160, reduceby):
    #        square.fill(arr[sqr[y][x]])
    #        draw_me=pygame.Rect((x*aspect+1), (y*aspect+1), reduceby * aspect, reduceby * aspect)
    #        screen.blit(square,draw_me)
    #  screen.blit(square,(0,0))
    #  pygame.display.flip()    


#def play_randomly(n):
#  screens = []
#  for episode in xrange(n):
#    total_reward = 0


#    while not ale.game_over():         
#      a = legal_actions[randrange(len(legal_actions))]
#      ale.act(a)
#      screens.append(ale.getScreen())
#    ale.reset_game()  
#  return screens  

#def define_background(screens):
#  import scipy.stats
#  return scipy.stats.mode(screens).mode    

#def pickle_background(background, file_name):
#  import pickle
#  with open(file_name, 'wb') as pfile:
#    pickle.dump(background, pfile, protocol = pickle.HIGHEST_PROTOCOL)  

#def unpickle_backround(file_name):
#  import pickle
#  with open(file_name, 'r') as pfile:
#    return pickle.load(pfile)    

# Play 10 episodes
#def play():
# for episode in xrange(10):
#  total_reward = 0


#  while not ale.game_over():     
#    import time
    #time.sleep(0.05)
    #show(sqr = np.reshape(ale.getScreen(), (210, 160)))
#    show_vectorized(vectorized(ale.getScreen()))
#    a = legal_actions[randrange(len(legal_actions))]
    # Apply an action and get the resulting reward
#    reward = ale.act(a);
#    total_reward += reward
#  print 'Episode', episode, 'ended with score:', total_reward
#  ale.reset_game()