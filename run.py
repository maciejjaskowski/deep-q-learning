import teacher as q
import ale_game as ag
import dqn
import theano
import lasagne
import network
import simple_breakout


def main(**kargs):

    if kargs['theano_verbose']:
        theano.config.compute_test_value = 'warn'
        theano.config.exception_verbosity = 'high'
        theano.config.optimizer = 'fast_compile'

    if kargs['game'] == 'simple_breakout':
        game = simple_breakout.SimpleBreakout()
        class P(object):
            def __init__(self):
                self.screen_size = (12, 12)

            def __call__(self, frames):
                return frames
        phi = P()
    else:
        ale = ag.init(game=kargs['game'], display_screen=(kargs['visualize'] == 'ale'), record_dir=kargs['record_dir'])
        game = ag.ALEGame(ale)
        phi = ag.Phi()

    replay_memory = dqn.ReplayMemory(size=kargs['replay_memory_size']) if not kargs['no_replay'] else None

    def create_algo():
        algo = dqn.DQNAlgo(game.n_actions(),
                               replay_memory=replay_memory,
                               build_network=kargs['network'],
                               updates=kargs['updates'],
                               screen_size=phi.screen_size)

        algo.replay_start_size = kargs['replay_start_size']
        algo.final_epsilon = kargs['final_epsilon']
        algo.initial_epsilon = kargs['initial_epsilon']

        algo.log_frequency = kargs['log_frequency']
        algo.target_network_update_frequency = kargs['target_network_update_frequency']
        algo.final_exploration_frame = kargs['final_exploration_frame']
        return algo

    algo_train = create_algo()
    algo_test = create_algo()
    algo_test.final_epsilon = 0.1
    algo_test.initial_epsilon = 0.1
    algo_test.epsilon = 0.1


    import Queue
    algo_train.mood_q = Queue.Queue() if kargs['show_mood'] else None

    if kargs['show_mood'] is not None:
        plot = kargs['show_mood']()

        def worker():
            while True:
                item = algo_train.mood_q.get()
                plot.show(item)
                algo_train.mood_q.task_done()

        import threading
        t = threading.Thread(target=worker)
        t.daemon = True
        t.start()

    print(str(algo_train))

    if kargs['visualize'] != 'q':
        visualizer = q.GameNoVisualizer()
    else:
        if kargs['game'] == 'simple_breakout':
            visualizer = simple_breakout.SimpleBreakoutVisualizer(algo_train)
        else:
            visualizer = ag.ALEGameVisualizer(phi.screen_size)

    teacher = q.Teacher(game=game,
                        algo=algo_train,
                        game_visualizer=visualizer,
                        phi=phi,
                        repeat_action=kargs['repeat_action'],
                        max_actions_per_game=kargs['max_actions_per_game'],
                        skip_n_frames_after_lol=kargs['skip_n_frames_after_lol'],
                        tester=False)

    tester = q.Teacher(game=game,
                        algo=algo_test,
                        game_visualizer=visualizer,
                        phi=phi,
                        repeat_action=kargs['repeat_action'],
                        max_actions_per_game=kargs['max_actions_per_game'],
                        skip_n_frames_after_lol=kargs['skip_n_frames_after_lol'],
                        tester=True)

    q.teach_and_test(teacher, tester, n_epochs=20,
                     frames_to_test_on=150000,
                     epoch_size=50000,
                     state_dir=kargs['weights_dir'],
                     algo_initial_state_file=kargs['algo_initial_state_file'])


class Log(object):
    def __init__(self):
        pass

    def show(self, info):
        print(str(info['i_frame']) + " | Expectations: " + str(info['expectations']))
        print(str(info['i_frame']) + " | Surprise: " + str(info['surprise']))


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

            self.fig.canvas.draw()
            self.fig.canvas.flush_events()



d = {
    'game': 'space_invaders',
    'reshape': 'max',
    'visualize': False,
    'record_dir': None,
    'weights_dir': 'weights',
    'theano_verbose': False,
    'show_mood': None,
    'replay_start_size': 50000,
    'initial_epsilon': 1,
    'final_epsilon': 0.1,
    'log_frequency': 1,
    'replay_memory_size': 400000,
    'no_replay': False,
    'network': network.build_nature,
    'updates': lambda loss, params: updates.deepmind_rmsprop(loss, params, learning_rate=.00025, rho=.95, epsilon=.01),
    'repeat_action': 4,
    'skip_n_frames_after_lol': 30,
    'target_network_update_frequency': 10000,
    'final_exploration_frame': 1000000,
    'algo_initial_state_file': None,
    'max_actions_per_game': 10000,
     }

if __name__ == "__main__":
    import sys
    import getopt
    optlist, args = getopt.getopt(sys.argv[1:], '', [
        'game=',
        'reshape=',
        'visualize=',
        'record_dir=',
        'replay_start_size=',
        'final_epsilon=',
        'initial_epsilon=',
        'log_frequency=',
        'replay_memory_size=',
        'theano_verbose=',
        'weights_dir=',
        'show_mood=',
        'no_replay',
        'network=',
        'updates='])

    for o, a in optlist:
        if o in ("--visualize",):
            d['visualize'] = a
        elif o in ("--game",):
            d['game'] = a
        elif o in ("--reshape",):
            d['reshape'] = a
        elif o in ("--record_dir",):
            d['record_dir'] = a
        elif o in ("--weights_dir",):
            d['weights_dir'] = a
        elif o in ("--replay_start_size",):
            d["replay_start_size"] = int(a)
        elif o in ("--final_epsilon",):
            d["final_epsilon"] = float(a)
        elif o in ("--initial_epsilon",):
            d["initial_epsilon"] = float(a)
            d["epsilon"] = float(a)
        elif o in ("--log_frequency",):
            d["log_frequency"] = int(a)
        elif o in ("--replay_memory_size",):
            d["replay_memory_size"] = int(a)
        elif o in ("--theano_verbose",):
            d["theano_verbose"] = bool(a)
        elif o in ("--show_mood",):
            if a == 'plot':
                d["show_mood"] = Plot
            elif a == "log":
                d["show_mood"] = Log
        elif o in ("--no_replay",):
            d["no_replay"] = True
        elif o in ("--network",):
            if a == 'nature':
                d["network"] = network.build_nature
            elif a == 'nature_with_pad':
                d["network"] = network.build_nature_with_pad
            elif a == 'nips':
                d["network"] = network.build_nips
            elif a == 'nature_with_pad_he':
                d["network"] = network.build_nature_with_pad_he
        elif o in ("--updates",):
            import updates
            if a == 'deepmind_rmsprop':
                d["updates"] = \
                    lambda loss, params: updates.deepmind_rmsprop(loss, params, learning_rate=.00025, rho=.95, epsilon=.01)
            elif a == 'rmsprop':
                d["updates"] = \
                    lambda loss, params: lasagne.updates.rmsprop(loss, params, learning_rate=.0002, rho=.95, epsilon=1e-6)
        else:
            assert False, "unhandled option"

    import pprint
    pp = pprint.PrettyPrinter(depth=2)
    print(optlist)
    print(args)
    print(sys.argv)
    print("")
    pp.pprint(d)

    main(**d)

