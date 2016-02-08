
import q_learning as q
import numpy as np
import ale_game as ag

def random_on_space_invaders():
  import q_learning as q
  import numpy as np
  import ale_game as ag
  reload(q)
  reload(ag)
  ale = ag.init()
  game = ag.SpaceInvadersGame(ale)
  #game.show_vectorized(game.vectorized(ale.getScreen()))
  teacher = q.Teacher(game, q.RandomAlgo(game.get_actions()), ag.SpaceInvadersGameVectorizedVisualizer())
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
    initial_q = 5, initial_theta = [1] * 14 * 20 * n_colors, be_positive = False, state_adapter = state_adapter)
  q_algo1.epsilon = 0.05
  q_algo1.lmbda = 0.99 # 0.9
  q_algo1.gamma = 0.999
  q_algo1.alpha = 0.5
  def new_game():
    game.ale.reset_game()
    game.finished = False
    game.cum_reward = 0
    return game

  teacher = q.Teacher(new_game, q_algo1, ag.SpaceInvadersGameVectorizedVisualizer(), repeat_action = 3)
  
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

  teacher = q.Teacher(new_game, q_algo1, q.GameNoVisualizer(), repeat_action = 3)
  q_algo1.log_freq = 0.05
  q_algo1.epsilon = 1  
  result_1 = teacher.teach(initial_training)


  q_algo1.epsilon = 0
  q_algo1.log_freq = 0.05
  result_test.append(teacher.teach(1))

  for i in range(training_decay_from):
    q_algo1.epsilon = 1 - i/100
    teacher = q.Teacher(new_game, q_algo1, q.GameNoVisualizer(), repeat_action = 3)
    result_2.append(teacher.teach(training_decay_ex))
    q_algo1.epsilon = 0
    result_test.append(teacher.teach(1))  


  import cPickle as pickle
  with open('gradient_descent.theta' + run , 'wb') as handle:
    pickle.dump(q_algo1.theta, handle)

  with open('gradient_descent.gamma' + run, 'wb') as handle:
    pickle.dump(q_algo1.gamma, handle)

  with open('gradient_descent.lmbda' + run, 'wb') as handle:
    pickle.dump(q_algo1.lmbda, handle)

  with open('gradient_descent.alpha' + run, 'wb') as handle:
    pickle.dump(q_algo1.alpha, handle)  

  r1 = [a[1] for a in result_1]  
  plt.plot(np.array([x[1] - x[0] for x in zip(np.cumsum(r1), np.cumsum(r1)[200:])])/200)

  r2 = [a[1] for r in result_2 for a in r]  
  plt.plot(np.array([x[1] - x[0] for x in zip(np.cumsum(r2), np.cumsum(r2)[200:])])/200)

  r_test = [a[1] for r in result_test for a in r]
  plt.plot(np.array([x[1] - x[0] for x in zip(np.cumsum(r_test), np.cumsum(r_test)[50:])])/50)

  r_4 = [a[1] for a in result_4 ]
  plt.plot(np.array([x[1] - x[0] for x in zip(np.cumsum(r_test), np.cumsum(r_4)[2:])])/2)
  

  q_algo1.epsilon = 0.1
  teacher = q.Teacher(new_game, q_algo1, q.GameNoVisualizer(), repeat_action = 3)
  teacher.teach(100)

  
def random_on_mountain_car_game():
  game = q.MountainCarGame()
  q_algo = q.RandomAlgo(game.get_actions())
  visualizer = q.MountainCarGameVisualizer()

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


  #dot = sum(initial_theta[phi((2,2))])


  state_adapter = q.mountain_car_game_tilings_state_adapter(n_tilings, tile_in_row)

  state_adapter2 = lambda s: np.array(state_adapter(s))

  initial_theta = np.array([1] * tile_in_row * tile_in_row * n_tilings)

  q_algo1 = ss.SARSALambdaGradientDescent(game().get_actions(), game().get_state(), 
    initial_q = 0, initial_theta = initial_theta, state_adapter = state_adapter2)

  q_algo1.epsilon = 0.02
  q_algo1.lmbda = 0.5
  q_algo1.gamma = 0.9
  q_algo1.alpha = 0.1
  
  teacher = q.Teacher(game, q_algo1, g.MountainCarGameVisualizer(q_algo1))
  teacher.teach(1)

  teacher = q.Teacher(game, q_algo1, q.GameNoVisualizer())
  teacher.teach(30)
  

class Tester:
  def test(self, game_factory, algo_factory, teach_rounds = 100, repeat = 100):
    return np.array([q.Teacher(game_factory, algo_factory(), q.GameNoVisualizer()).teach(teach_rounds) 
      for i in range(0, repeat)]).mean(axis = 0)


def test():
  import q_learning as q
  import numpy as np
  reload(q)
  game = game_collect_all
  
  def algo_prio():
    memory = q.PrioritizedMemory(100, 0)
    algo = q.SARSALambdaPrioritizedMemory(game().get_actions(), memory)
    algo.epsilon = 0.1
    algo.gamma = 0.7
    algo.alpha = 0.1
    return algo

  def algo_sarsa_lambda():
    q_algo1 = q.SARSALambda(game().get_actions(), game().get_state(), 20, 4)
    q_algo1.lmbda = 0.8
    q_algo1.gamma = 0.7
    q_algo1.alpha = 0.1
    q_algo1.epsilon = 0.1
    return q_algo1

  result_prio = Tester().test(game, algo_prio, teach_rounds = 60, repeat = 50)
  result_sarsa_lambda = Tester().test(game, algo_sarsa_lambda, teach_rounds = 60, repeat = 50)

        

def teach_off_repeat():
  def factory(game):
    q_algo1 = q.SARSRepeat(game.get_actions(), game.get_state(), sample_size = 1, history_length = 20)
    q_algo1.gamma = 0.9


    q_algo1.alpha = 0.1
    q_algo1.epsilon = 0.1
    return q_algo1
  return Tester().test(game_collect_all, factory)
  
def teach_off():
  def factory(game):
    q_algo1 = q.SARSRepeat(game.get_actions(), game.get_state())
    q_algo1.gamma = 0.9

    q_algo1.alpha = 0.1
    q_algo1.epsilon = 0.1
    return q_algo1
  return Tester().test(game_collect_all, factory)


  # compare variation between repeat and not-repeat: plt.plot(map(lambda x: np.concatenate([x[0], x[1]], axis = 0), zip(u, t)))

