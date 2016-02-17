def random_on_space_invaders():
    import q_learning as q
    import ale_game as ag
    reload(q)
    reload(ag)

    ale = ag.init()
    game = ag.SpaceInvadersGame(ale)

    def new_game():
        game.ale.reset_game()
        game.finished = False
        game.cum_reward = 0
        return game

    # game.show_vectorized(game.vectorized(ale.getScreen()))
    teacher = q.Teacher(new_game, q.RandomAlgo(game.get_actions()), ag.SpaceInvadersGameCombined2Visualizer(),
                        ag.Phi(skip_every=6), repeat_action=6)
    teacher.teach(1)


def dqn_on_space_invaders(visualize=False, theano_verbose=False, initial_weights_file=None, ignore_feedback=False):
    import q_learning as q
    import ale_game as ag
    import dqn
    import theano
    reload(q)
    reload(ag)
    reload(dqn)
    if theano_verbose:
        theano.config.compute_test_value = 'warn'
        theano.config.exception_verbosity = 'high'
        theano.config.optimizer = 'fast_compile'

    ale = ag.init()
    game = ag.SpaceInvadersGame(ale)

    def new_game():
        game.ale.reset_game()
        game.finished = False
        game.cum_reward = 0
        game.lives = 4
        return game

    replay_memory = dqn.ReplayMemory(size=100, grace=10)
    dqn_algo = dqn.DQNAlgo(game.n_actions(), replay_memory=replay_memory, initial_weights_file=initial_weights_file)

    dqn_algo.target_network_update_frequency = 50
    dqn_algo.replay_memory_size = 100
    dqn_algo.replay_start_size = 75
    dqn_algo.epsilon = 0.1
    dqn_algo.initial_epsilon = 0.1
    dqn_algo.final_epsilon = 0.1

    dqn_algo.ignore_feedback = ignore_feedback
    # dqn_algo.ignore_feedback = True

    visualizer = ag.SpaceInvadersGameCombined2Visualizer() if visualize else q.GameNoVisualizer()
    teacher = q.Teacher(new_game, dqn_algo, visualizer,
                        ag.Phi(skip_every=4), repeat_action=4, sleep_seconds=0)
    teacher.teach(500000)


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


def sarsa_gd_on_space_invaders():
    import q_learning as q
    import numpy as np
    import ale_game as ag
    import matplotlib.pyplot as plt
    import sarsa as ss

    plt.ion()
    reload(ss)
    reload(q)
    reload(ag)
    ale = ag.init()
    run = '1'

    def state_adapter(frames):
        result = np.where(np.reshape(np.concatenate(frames), 80 * 80 * 4) > 0)
        if len(result) == 0:
            return [0]
        else:
            return result

    game = ag.SpaceInvadersGame(ale)
    q_algo1 = ss.SARSALambdaGradientDescent(game.n_actions(), theta_len=80 * 80 * 4, state_adapter=state_adapter)
    q_algo1.epsilon = 0.9
    q_algo1.lmbda = 0.99
    q_algo1.gamma = 0.999
    q_algo1.alpha = 0.1

    def new_game():
        game.ale.reset_game()
        game.finished = False
        game.cum_reward = 0
        return game



    result_test = []
    result_1 = []
    result_2 = []

    teacher = q.Teacher(new_game, q_algo1, q.GameNoVisualizer(), phi=ag.Phi(skip_every=6), repeat_action=6)

    q_algo1.epsilon = 1
    q_algo1.log_freq = 1
    result_test.append(teacher.teach(10))

    vis_teacher = q.Teacher(new_game, q_algo1, ag.SpaceInvadersGameCombined2Visualizer(), phi=ag.Phi(skip_every=6),
                        repeat_action=6)

    #  teacher.single_step(Game)
    q_algo1.epsilon = 0.1
    q_algo1.log_freq = 1
    # vis_teacher.teach(5)

    for i in xrange(90):
        q_algo1.log_freq = 0.03
        q_algo1.epsilon = 1 - i / 100
        result_2.append(teacher.teach(50))
        q_algo1.epsilon = 0.1
        result_test.append(teacher.teach(10))

    import cPickle as pickle
    with open('gradient_descent.theta' + run, 'wb') as handle:
        pickle.dump(q_algo1.theta, handle)

    with open('gradient_descent.gamma' + run, 'wb') as handle:
        pickle.dump(q_algo1.gamma, handle)

    with open('gradient_descent.lmbda' + run, 'wb') as handle:
        pickle.dump(q_algo1.lmbda, handle)

    with open('gradient_descent.alpha' + run, 'wb') as handle:
        pickle.dump(q_algo1.alpha, handle)

    r1 = [a[1] for a in result_1]
    plt.plot(np.array([x[1] - x[0] for x in zip(np.cumsum(r1), np.cumsum(r1)[200:])]) / 200)

    r2 = [a[1] for r in result_2 for a in r]
    plt.plot(np.array([x[1] - x[0] for x in zip(np.cumsum(r2), np.cumsum(r2)[200:])]) / 200)

    r_test = [a[1] for r in result_test for a in r]
    plt.plot(np.array([x[1] - x[0] for x in zip(np.cumsum(r_test), np.cumsum(r_test)[50:])]) / 50)

    r_4 = [a[1] for a in result_4]
    plt.plot(np.array([x[1] - x[0] for x in zip(np.cumsum(r_test), np.cumsum(r_4)[2:])]) / 2)

    q_algo1.epsilon = 0.1
    teacher = q.Teacher(new_game, q_algo1, q.GameNoVisualizer(), repeat_action=3)
    teacher.teach(100)


def random_on_mountain_car_game():
    import games as g
    import q_learning as q
    reload(q)
    reload(g)
    game = g.MountainCarGame()
    q_algo = q.RandomAlgo(game.get_actions())
    visualizer = g.MountainCarGameVisualizer()

    teacher = q.Teacher(game, q_algo, visualizer)

    teacher.teach(1)


import getopt
import sys

try:
    opts = getopt.getopt(sys.argv, "vw:", ["visualize", "weights"])
except getopt.GetoptError:
    print("wrong parameters")
    sys.exit(2)
#
# visualize = False
# initial_weights_file = None
# for opt, arg in opts:
#     if opt in ("-v", "--visualize"):
#         visualize = True
#     elif opt in ("-w", "--weights"):
#         initial_weights_file = arg

#dqn_on_space_invaders(visualize=visualize, initial_weights_file=initial_weights_file)
dqn_on_space_invaders(visualize=True, initial_weights_file='weights_2400100.npz', ignore_feedback=True)
