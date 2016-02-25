#def dqn_on_space_invaders_gpu(visualize=False, theano_verbose=False, initial_weights_file=None, initial_i_frame=0):
import q_learning as q
import ale_game as ag
import dqn
import theano


def latest(dir='.'):
    import os, re
    frames = [int(re.match(r"weights_([0-9]*).npz", file).groups()[0])
             for file in os.listdir(dir) if file.startswith("weights_")]
    if frames == None or len(frames) == 0:
        return None, 0
    else:
        return dir + '/weights_' + str(max(frames)) + '.npz', max(frames)


initial_weights_file, initial_i_frame = latest("weights")
theano_verbose = False
replay_memory_size = 500000
visualize = False


print("Using weights from file: ", initial_weights_file)

if theano_verbose:
    theano.config.compute_test_value = 'warn'
    theano.config.exception_verbosity = 'high'
    theano.config.optimizer = 'fast_compile'

ale = ag.init(display_screen=(visualize == 'ale'))
game = ag.SpaceInvadersGame(ale)


def new_game():
    game.ale.reset_game()
    game.finished = False
    game.cum_reward = 0
    game.lives = 4
    return game

#replay_memory = dqn.ReplayMemory(size=100, grace=10)
replay_memory = dqn.ReplayMemory(size=replay_memory_size)
dqn_algo = dqn.DQNAlgo(game.n_actions(), replay_memory=replay_memory, initial_weights_file=initial_weights_file,
                       build_cnn=dqn.build_cnn_gpu)

if initial_weights_file:
    dqn_algo.replay_start_size = 250000
    dqn_algo.epsilon = 0.1
    dqn_algo.final_epsilon = 0.1
    dqn_algo.initial_epsilon = 0.1
    dqn_algo.i_frames = initial_i_frame

dqn_algo.log_frequency=1

# dqn_algo.target_network_update_frequency = 50
# dqn_algo.replay_memory_size = 100
# dqn_algo.replay_start_size = 75
# dqn_algo.epsilon = 0.1
# dqn_algo.initial_epsilon = 0.1
# dqn_algo.final_epsilon = 0.1
# dqn_algo.log_frequency = 10
# visualize = True
# dqn_algo.ignore_feedback = ignore_feedback


# dqn_algo.epsilon = 0.1
# dqn_algo.initial_epsilon = 0.1
# dqn_algo.final_epsilon = 0.1
# dqn_algo.ignore_feedback = True
# dqn_algo.log_frequency = 0
#
# import Queue
# dqn_algo.mood_q = Queue.Queue() if show_mood else None
#
# if show_mood:
#     plot = Plot()
#
#     def worker():
#         while True:
#             item = dqn_algo.mood_q.get()
#             plot.show(item)
#             dqn_algo.mood_q.task_done()
#
#     import threading
#     t = threading.Thread(target=worker)
#     t.daemon = True
#     t.start()

print(str(dqn_algo))

visualizer = ag.SpaceInvadersGameCombined2Visualizer() if visualize == 'q' else q.GameNoVisualizer()
teacher = q.Teacher(new_game, dqn_algo, visualizer,
                    ag.Phi(skip_every=4), repeat_action=4, sleep_seconds=0)
teacher.teach(500000)


class Plot(object):

    def __init__(self):
        import matplotlib.pyplot as plt
        plt.ion()
        self.fig = plt.figure()
        plt.title('Surprise')
        plt.ylabel('Surprise (red), Expectation (blue)')
        plt.xlabel('frame')

        self.expectation = self.fig.add_subplot(2, 1, 1)
        self.expectations_l, = self.expectation.plot([], [], color='b', linestyle='-', lw=2)
        self.expectation.set_xlim([0, 105])
        self.expectation.set_ylim([-5, 10])

        self.surprise = self.fig.add_subplot(2, 1, 2)
        self.surprise_l, = self.surprise.plot([], [], color='r', linestyle='-', lw=2)
        self.surprise.set_xlim([0, 105])
        self.surprise.set_ylim([-5, 5])

        self.expectations_y = []
        self.surprise_y = []
        self.i = 0
        self.print_every_n = 1

    def show(self, info):
        self.i += 1

        self.expectations_y.append(info['expectations'])
        self.surprise_y.append(info['surprise'])
        if len(self.expectations_y) > 100:
            self.expectations_y = self.expectations_y[1:]
            self.surprise_y = self.surprise_y[1:]

        print(info)
        if self.i % self.print_every_n == 0:
            self.expectations_l.set_xdata(list(range(len(self.expectations_y))))
            self.expectations_l.set_ydata(self.expectations_y)
            self.surprise_l.set_xdata(list(range(len(self.surprise_y))))
            self.surprise_l.set_ydata(self.surprise_y)

            # self.mi.set_xdata([self.xlim[0]] * 2)
            # self.mi.set_ydata([-0.08, 0.08])
            # self.ma.set_xdata([self.xlim[1]] * 2)
            # self.ma.set_ydata([-0.08, 0.08])

            #pos = np.arange(-1.2, 0.5, 0.05)
            #vel = np.arange(-0.07, 0.07, 0.005)

            #self.expectation.line(expectations)

            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

            #time.sleep(0.01)


def const_on_space_invaders():
    import q_learning as q
    import ale_game as ag
    import dqn
    reload(q)
    reload(ag)
    reload(dqn)

    ale = ag.init()
    game = ag.SpaceInvadersGame(ale)

    def new_game():
        game.ale.reset_game()
        game.finished = False
        game.cum_reward = 0
        return game

    const_algo = q.ConstAlgo([2, 2, 2, 2, 2, 0, 0, 0, 0])
    teacher = q.Teacher(new_game, const_algo, ag.SpaceInvadersGameCombined2Visualizer(),
                        ag.Phi(skip_every=6), repeat_action=6)
    teacher.teach(1)





#dqn_on_space_invaders(visualize=visualize, initial_weights_file=initial_weights_file)
#dqn_on_space_invaders(visualize=True, initial_weights_file='weights_2400100.npz', ignore_feedback=True)


#dqn_on_space_invaders_gpu(visualize=False, initial_weights_file=latest('.')[0], initial_i_frame=latest('.')[1])

#results = dqn_on_space_invaders_play(visualize=None, initial_weights_file='analysis/sth_working_900000.npz', show_mood=False)


#results = dqn_on_space_invaders_cpu(visualize=None, initial_weights_file=None)
#results = dqn_on_space_invaders_play(visualize='ale', initial_weights_file=latest('analysis')[0], show_mood=True)
#results = dqn_on_space_invaders_play(visualize='q', initial_weights_file=None, show_mood=True)
#results = dqn_on_space_invaders_play(visualize='ale', initial_weights_file='analysis/weights_800100.npz', show_mood=True)
#results = dqn_on_space_invaders_play(visualize='q', initial_weights_file='analysis/weights_1000100.npz')
#results = dqn_on_space_invaders_play(visualize='q', initial_weights_file='analysis/weights_800100.npz')

#pickle.dump(results, open("results_900000_new.pickled", "wb"))
