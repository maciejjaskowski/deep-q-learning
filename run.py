import teacher as q
import ale_game as ag
import dqn
import theano
import lasagne
import network
import simple_breakout
import updates as u
import pprint


def main(game_name, network_type, updates_method,
         target_network_update_frequency,
         initial_epsilon, final_epsilon, test_epsilon, final_exploration_frame, replay_start_size,
         deepmind_rmsprop_epsilon, deepmind_rmsprop_learning_rate, deepmind_rmsprop_rho,
         rmsprop_epsilon, rmsprop_learning_rate, rmsprop_rho,
         phi_type, phi_method,
         epoch_size, n_training_epochs, n_test_epochs,
         visualize, record_dir, show_mood,
         replay_memory_size, no_replay,
         repeat_action, skip_n_frames_after_lol, max_actions_per_game,
         weights_dir, algo_initial_state_file,
         log_frequency, theano_verbose):
    args = locals()

    if theano_verbose:
        theano.config.compute_test_value = 'warn'
        theano.config.exception_verbosity = 'high'
        theano.config.optimizer = 'fast_compile'

    if game_name == 'simple_breakout':
        game = simple_breakout.SimpleBreakout()
        class P(object):
            def __init__(self):
                self.screen_size = (12, 12)

            def __call__(self, frames):
                return frames
        phi = P()
    else:
        ale = ag.init(game=game_name, display_screen=(visualize == 'ale'), record_dir=record_dir)
        game = ag.ALEGame(ale)
        if phi_type == '4':
            phi = ag.Phi4(method=phi_method)
        elif phi_type == '1':
            phi = ag.Phi(method=phi_method)
        else:
            raise RuntimeError("Unknown phi: {phi}".format(phi=phi_type))

    if network_type == 'nature':
        build_network = network.build_nature
    elif network_type == 'nature_with_pad':
        build_network = network.build_nature_with_pad
    elif network_type == 'nips':
        build_network = network.build_nips
    elif network_type == 'nature_with_pad_he':
        build_network = network.build_nature_with_pad_he
    elif hasattr(network_type, '__call__'):
        build_network = network_type
    else:
        raise RuntimeError("Unknown network: {network}".format(network=network_type))


    if updates_method == 'deepmind_rmsprop':
        updates = \
            lambda loss, params: u.deepmind_rmsprop(loss, params,
                                                          learning_rate=deepmind_rmsprop_learning_rate,
                                                          rho=deepmind_rmsprop_rho,
                                                          epsilon=deepmind_rmsprop_epsilon)
    elif updates_method == 'rmsprop':
        updates = \
            lambda loss, params: lasagne.updates.rmsprop(loss, params,
                                                         learning_rate=rmsprop_learning_rate,
                                                         rho=rmsprop_rho,
                                                         epsilon=rmsprop_epsilon)
    else:
        raise RuntimeError("Unknown updates: {updates}".format(updates=updates_method))

    replay_memory = dqn.ReplayMemory(size=replay_memory_size) if not no_replay else None

    def create_algo():
        algo = dqn.DQNAlgo(game.n_actions(),
                               replay_memory=replay_memory,
                               build_network=build_network,
                               updates=updates,
                               screen_size=phi.screen_size)

        algo.replay_start_size = replay_start_size
        algo.final_epsilon = final_epsilon
        algo.initial_epsilon = initial_epsilon

        algo.log_frequency = log_frequency
        algo.target_network_update_frequency = target_network_update_frequency
        algo.final_exploration_frame = final_exploration_frame
        return algo

    algo_train = create_algo()
    algo_test = create_algo()
    algo_test.final_epsilon = test_epsilon
    algo_test.initial_epsilon = test_epsilon
    algo_test.epsilon = test_epsilon


    import Queue
    algo_train.mood_q = Queue.Queue() if show_mood else None

    if show_mood is not None:
        import Queue
        algo_train.mood_q = Queue.Queue()
        if show_mood == 'plot':
            plot = Plot()
        elif show_mood == "log":
            plot = Log()

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

    if visualize != 'q':
        visualizer = q.GameNoVisualizer()
    else:
        if game_name == 'simple_breakout':
            visualizer = simple_breakout.SimpleBreakoutVisualizer(algo_train)
        else:
            visualizer = ag.ALEGameVisualizer(phi.screen_size)

    teacher = q.Teacher(game=game,
                        algo=algo_train,
                        game_visualizer=visualizer,
                        phi=phi,
                        repeat_action=repeat_action,
                        max_actions_per_game=max_actions_per_game,
                        skip_n_frames_after_lol=skip_n_frames_after_lol,
                        tester=False)

    tester = q.Teacher(game=game,
                        algo=algo_test,
                        game_visualizer=visualizer,
                        phi=phi,
                        repeat_action=repeat_action,
                        max_actions_per_game=max_actions_per_game,
                        skip_n_frames_after_lol=skip_n_frames_after_lol,
                        tester=True)

    q.teach_and_test(teacher, tester, n_epochs=n_training_epochs,
                     frames_to_test_on=n_test_epochs * epoch_size,
                     epoch_size=epoch_size,
                     state_dir=weights_dir,
                     algo_initial_state_file=algo_initial_state_file)


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



defaults = {
    'game_name': 'space_invaders',
    'visualize': False,
    'record_dir': None,
    'weights_dir': 'weights',
    'theano_verbose': False,
    'show_mood': None,
    'replay_start_size': 50000,
    'initial_epsilon': 1,
    'final_epsilon': 0.1,
    'test_epsilon': 0.05,
    'log_frequency': 1,
    'replay_memory_size': 400000,
    'no_replay': False,
    'network_type': 'nature',
    'updates_method': 'rmsprop',
    'repeat_action': 4,
    'skip_n_frames_after_lol': 30,
    'target_network_update_frequency': 10000,
    'final_exploration_frame': 1000000,
    'algo_initial_state_file': None,
    'max_actions_per_game': 10000,
    'phi_type': '1',
    'phi_method': 'resize',

    'deepmind_rmsprop_learning_rate': 0.00025,
    'deepmind_rmsprop_rho': .95,
    'deepmind_rmsprop_epsilon': 0.01,
    'rmsprop_learning_rate': 0.0002,
    'rmsprop_rho': .95,
    'rmsprop_epsilon': 1e-6,

    'n_training_epochs': 50,
    'n_test_epochs': 1,
    'epoch_size': 50000
     }


