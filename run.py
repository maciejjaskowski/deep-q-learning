import teacher as q
import ale_game as ag
import dqn
import theano
import lasagne
import network
import simple_breakout

def latest(dir='.'):
    if dir == None:
        return None, 0
    import os, re
    frames = [int(re.match(r"weights_([0-9]*).npz", file).groups()[0])
             for file in os.listdir(dir) if file.startswith("weights_")]
    if frames == None or len(frames) == 0:
        return None, 0
    else:
        return dir + '/weights_' + str(max(frames)) + '.npz', max(frames)


def main(**kargs):
    initial_weights_file, i_total_action = latest(kargs['weights_dir'])

    print("Continuing using weights from file: ", initial_weights_file, "from", i_total_action)

    if kargs['theano_verbose']:
        theano.config.compute_test_value = 'warn'
        theano.config.exception_verbosity = 'high'
        theano.config.optimizer = 'fast_compile'

    if kargs['game'] == 'simple_breakout':
        game = simple_breakout.SimpleBreakout()
        class P(object):
            def __init__(self):
                self.screen_size = 12

            def __call__(self, frames):
                return frames
        phi = P()
    else:
        ale = ag.init(game=kargs['game'], display_screen=(kargs['visualize'] == 'ale'), record_dir=kargs['record_dir'])
        game = ag.ALEGame(ale)
        phi = ag.Phi()

    replay_memory = dqn.ReplayMemory(size=kargs['dqn.replay_memory_size']) if not kargs['dqn.no_replay'] else None
    algo = dqn.DQNAlgo(game.n_actions(),
                           replay_memory=replay_memory,
                           initial_weights_file=initial_weights_file,
                           build_network=kargs['dqn.network'],
                           updates=kargs['dqn.updates'],
                           screen_size=phi.screen_size)

    algo.replay_start_size = kargs['dqn.replay_start_size']
    algo.final_epsilon = kargs['dqn.final_epsilon']
    algo.initial_epsilon = kargs['dqn.initial_epsilon']
    algo.i_action = i_total_action

    algo.log_frequency = kargs['dqn.log_frequency']
    algo.target_network_update_frequency = kargs['target_network_update_frequency']


    import Queue
    algo.mood_q = Queue.Queue() if kargs['show_mood'] else None

    if kargs['show_mood'] is not None:
        plot = kargs['show_mood']()

        def worker():
            while True:
                item = algo.mood_q.get()
                plot.show(item)
                algo.mood_q.task_done()

        import threading
        t = threading.Thread(target=worker)
        t.daemon = True
        t.start()

    print(str(algo))

    if kargs['visualize'] != 'q':
        visualizer = q.GameNoVisualizer()
    else:
        if kargs['game'] == 'simple_breakout':
            visualizer = simple_breakout.SimpleBreakoutVisualizer(algo)
        else:
            visualizer = ag.ALEGameVisualizer(phi.screen_size)

    teacher = q.Teacher(game=game,
                        algo=algo,
                        game_visualizer=visualizer,
                        phi=phi,
                        repeat_action=kargs['repeat_action'],
                        i_total_action=i_total_action,
                        total_n_actions=50000000,
                        max_actions_per_game=10000,
                        skip_n_frames_after_lol=kargs['skip_n_frames_after_lol'])
    teacher.teach()


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
    'dqn.replay_start_size': 50000,
    'dqn.initial_epsilon': 1,
    'dqn.final_epsilon': 0.1,
    'dqn.log_frequency': 1,
    'dqn.replay_memory_size': 400000,
    'dqn.no_replay': False,
    'dqn.network': network.build_nature,
    'dqn.updates': lasagne.updates.rmsprop,
    'repeat_action': 4,
    'skip_n_frames_after_lol': 30,
    'target_network_update_frequency': 10000
     }

if __name__ == "__main__":
    import sys
    import getopt
    optlist, args = getopt.getopt(sys.argv[1:], '', [
        'game=',
        'reshape=',
        'visualize=',
        'record_dir=',
        'dqn.replay_start_size=',
        'dqn.final_epsilon=',
        'dqn.initial_epsilon=',
        'dqn.log_frequency=',
        'replay_memory_size=',
        'theano_verbose=',
        'weights_dir=',
        'show_mood=',
        'dqn.no_replay',
        'dqn.network=',
        'dqn.updates='])

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
        elif o in ("--dqn.replay_start_size",):
            d["replay_start_size"] = int(a)
        elif o in ("--dqn.final_epsilon",):
            d["dqn.final_epsilon"] = float(a)
        elif o in ("--dqn.initial_epsilon",):
            d["dqn.initial_epsilon"] = float(a)
            d["dqn.epsilon"] = float(a)
        elif o in ("--dqn.log_frequency",):
            d["dqn.log_frequency"] = int(a)
        elif o in ("--replay_memory_size",):
            d["replay_memory_size"] = int(a)
        elif o in ("--theano_verbose",):
            d["theano_verbose"] = bool(a)
        elif o in ("--show_mood",):
            if a == 'plot':
                d["show_mood"] = Plot
            elif a == "log":
                d["show_mood"] = Log
        elif o in ("--dqn.no_replay",):
            d["dqn.no_replay"] = True
        elif o in ("--dqn.network",):
            if a == 'nature':
                d["dqn.network"] = network.build_nature
            elif a == 'nature_with_pad':
                d["dqn.network"] = network.build_nature_with_pad
            elif a == 'nature_with_pad2':
                d["dqn.network"] = network.build_nature_with_pad2
            elif a == 'nature_with_pad3':
                d["dqn.network"] = network.build_nature_with_pad3
            elif a == 'nips':
                d["dqn.network"] = network.build_nips
            elif a == 'nature_dnn':
                d["dqn.network"] = network.build_nature_dnn
            elif a == 'nips_dnn':
                d["dqn.network"] = network.build_nips_dnn
        elif o in ("--dqn.updates",):
            import updates
            if a == 'deepmind_rmsprop':
                d["dqn.updates"] = \
                    lambda loss, params: updates.deepmind_rmsprop(loss, params, learning_rate=.00025, rho=.95, epsilon=.01)
            elif a == 'rmsprop':
                d["dqn.updates"] = \
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

