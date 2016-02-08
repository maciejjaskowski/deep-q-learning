Usage:
  > ipython
  > %run "q-learning"
  > q_algo = QLearningAlgo(['u', 'd', 'l', 'r'], board.get_state())
  > teacher = Teacher(board, q_algo)
  > teacher.teach(100)
