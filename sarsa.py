import numpy as np

class SARSALambdaGradientDescent: 

    def __init__(self, actions, init_state, initial_q, initial_theta, be_positive = True, state_adapter = lambda x: x):
      self.lmbda = 0.8
      self.gamma = 0.7
      self.alpha = 0.1
      self.epsilon = 0.1
      self.state_adapter = state_adapter
      self.be_positive = be_positive
      self.log_freq = 0.05
      self.game_over_regret = 0
      
      self.actions = actions
      self.action_ind = dict(zip(self.actions, range(len(self.actions))))
      self.state = init_state
      self.initial_q = initial_q
      if self.be_positive:
        self.visited = set()

      self.theta = np.zeros([len(self.actions), len(initial_theta)])
      
      self.e = np.zeros([len(self.actions), len(initial_theta)])

    def phi(self, state):
      return self.state_adapter(state)

    def q(self, state, action):
      return self._q(self.phi(state), action)  

    def _q(self, state, action):  
      return sum(self.theta[self.action_ind[action]][state]) 

    def q_positive(self, state, action):
      if ((not self.be_positive) or (tuple(self.phi(state)), action) in self.visited):
        return self.q(state, action)
      else:
        return self.initial_q  

    def action(self):
      self.next_action = self._action(self.state)
      while True:
        yield self.next_action

    def _action(self, state):     
      if (random() < self.epsilon):
        return choice(self.actions)
      else:
        return self.pi(state)
          
    def best_action(self, state):
      return self.pi(state)

    def pi(self, state):      
      return max(map(lambda action: (action, self.q_positive(state, action)), self.actions), 
                 key = lambda opt: opt[1])[0]

    def pi_value(self, state):
      return max(map(lambda action: (action, self.q_positive(state, action)), self.actions), 
                 key = lambda opt: opt[1])[1]

    def feedback(self, exp):
        a1 = self._action(exp.s1)
        s0 = self.phi(exp.s0)
        s1 = self.phi(exp.s1)
        r0 = exp.r0#max(min(exp.r0, 1), -1) - 0.1
        if self.be_positive:
          self.visited.add((tuple(s0), exp.a0))    

        if exp.game_over:
          r0 = self.game_over_regret

        delta = r0 + (1 - int(exp.game_over)) * (self.gamma * self._q(s1, a1) - self._q(s0, exp.a0))
        if (random() < self.log_freq):
          print ("game_over ", int(exp.game_over), "delta ", delta)
          print ("r", r0, "g", self.gamma, "q1", self._q(s1, a1), "q0", self._q(s0, exp.a0))
          print ("a0", "a1", exp.a0, a1)        
        
        self.theta += (self.alpha * delta) * self.e
        self.e *= self.gamma
        self.e *= self.lmbda
          
        
        for a in self.actions:
            self.e[self.action_ind[a]][s1] = 0
        self.e[self.action_ind[a1]][s1] = 1.0 / len(s1)     
        
        self.state = exp.s1
        self.next_action = a1               

