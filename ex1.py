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


def dqn_on_space_invaders():
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

    dqn_algo = dqn.DQNAlgo(game.get_actions())
    teacher = q.Teacher(new_game, dqn_algo, ag.SpaceInvadersGameCombined2Visualizer(),
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

    n_colors = 5

    def state_adapter(scr):
        vect = np.reshape(ag.vectorized(scr, 14, 20), 14 * 20 * n_colors)
        return np.where(vect)[0]

    game = ag.SpaceInvadersGame(ale)
    q_algo1 = ss.SARSALambdaGradientDescent(game.get_actions(), game.get_state(),
                                            initial_q=5, initial_theta=[1] * 14 * 20 * n_colors, be_positive=False,
                                            state_adapter=state_adapter)
    q_algo1.epsilon = 0.05
    q_algo1.lmbda = 0.99  # 0.9
    q_algo1.gamma = 0.999
    q_algo1.alpha = 0.5

    def new_game():
        game.ale.reset_game()
        game.finished = False
        game.cum_reward = 0
        return game

    teacher = q.Teacher(new_game, q_algo1, ag.SpaceInvadersGameVectorizedVisualizer(), repeat_action=3)

    #  teacher.single_step(Game)
    q_algo1.epsilon = 0
    q_algo1.log_freq = 1
    teacher.teach(1)

    initial_training = 1000
    training_decay_from = 95
    training_decay_ex = 50

    result_test = []
    result_1 = []
    result_2 = []

    teacher = q.Teacher(new_game, q_algo1, q.GameNoVisualizer(), repeat_action=3)
    q_algo1.log_freq = 0.05
    q_algo1.epsilon = 1
    result_1 = teacher.teach(initial_training)

    q_algo1.epsilon = 0
    q_algo1.log_freq = 0.05
    result_test.append(teacher.teach(1))

    for i in range(training_decay_from):
        q_algo1.epsilon = 1 - i / 100
        teacher = q.Teacher(new_game, q_algo1, q.GameNoVisualizer(), repeat_action=3)
        result_2.append(teacher.teach(training_decay_ex))
        q_algo1.epsilon = 0
        result_test.append(teacher.teach(1))

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


def sarsa_lambda_gradient_descent():
    import matplotlib.pyplot as plt
    plt.ion()
    import q_learning as q
    import numpy as np
    import sarsa as ss
    import games as g
    reload(g)
    reload(q)
    reload(ss)

    game = g.MountainCarGame

    tile_in_row = 9
    n_tilings = 5

    state_adapter = g.mountain_car_game_tilings_state_adapter(n_tilings, tile_in_row)

    state_adapter2 = lambda s: np.array(state_adapter(s))

    initial_theta = np.array([1] * tile_in_row * tile_in_row * n_tilings)

    q_algo1 = ss.SARSALambdaGradientDescent(game().get_actions(), game().get_state(),
                                            initial_q=0, initial_theta=initial_theta, state_adapter=state_adapter2)

    q_algo1.epsilon = 0.02
    q_algo1.lmbda = 0.5
    q_algo1.gamma = 0.9
    q_algo1.alpha = 0.1

    teacher = q.Teacher(game, q_algo1, g.MountainCarGameVisualizer(q_algo1))
    teacher.teach(1)

    teacher = q.Teacher(game, q_algo1, q.GameNoVisualizer())
    teacher.teach(30)

dqn_on_space_invaders()