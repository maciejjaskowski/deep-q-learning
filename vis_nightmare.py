#def dqn_on_space_invaders_gpu(visualize=False, theano_verbose=False, initial_weights_file=None, initial_i_frame=0):
import q_learning as q
import ale_game as ag
import dqn
import theano
import theano.tensor as T
import lasagne
import numpy as np


def vis():
    import os
    import ale_game as ag
    import pygame
    pygame.init()
    vis = ag.SpaceInvadersGameCombined2Visualizer()
    paths = sorted(['dream/' + f for f in os.listdir('dream')])

    for path in paths[1:len(paths):10]:
        with np.load(path) as screen:
            print(path)
            print(screen['arr_0'] * 256)
            vis.show(screen['arr_0'] * 256)
            import time
            time.sleep(3)

            #raw_input("Press Enter to continue...")


vis()
