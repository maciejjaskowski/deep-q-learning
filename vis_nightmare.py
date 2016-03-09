#def dqn_on_space_invaders_gpu(visualize=False, theano_verbose=False, initial_weights_file=None, initial_i_frame=0):
import q_learning as q
import ale_game as ag
import dqn
import theano
import theano.tensor as T
import lasagne
import numpy as np
import time

def vis():
    import os
    import ale_game as ag
    import pygame
    pygame.init()
    vis = ag.SpaceInvadersGameCombined2Visualizer()
    paths = sorted(['dream/' + f for f in os.listdir('dream')])

    for path in paths[0:len(paths):10]:
        with np.load(path) as screen:
            print(path)
            while True:
                for i in range(4):
                    vis.show(256 * np.stack([screen['arr_0'][0][i],
                              screen['arr_0'][0][i],
                              screen['arr_0'][0][i],
                              screen['arr_0'][0][i]]).reshape((1,4,80,80)))
                    time.sleep(0.3)

            #print(screen['arr_0'] * 256)
            #vis.show(screen['arr_0'] * 256)



            #raw_input("Press Enter to continue...")


vis()
