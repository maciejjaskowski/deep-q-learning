# Deep Q-Learning

## Overview

Our version of the deep q-learning algorithm from [The DQN
paper](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf). This algorithm reads
the screen and the integer score of the Atari 2600 game Space Invaders. The
output is the same control commands as a human would have with a controller
(albeit, without the physical controller).

## Installation Dependencies:
* Python 2.7
* Theano
* Lasagne
* pygame
* [Arcade Learning Environment (ALE) 0.5.1](arcadelearningenvironment.org)
* Atari 2600 ROM of space_invaders.bin


## External References

[The DQN paper](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)

[Human-level control through deep reinforcement
learning](https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf)

[Arcade Learning Environment](arcadelearningenvironment.org)


[Reccurent Model of Visual Attention](http://papers.nips.cc/paper/5542-recurrent-models-of-visual-attention.pdf) - applying q-learning to figure out what part of the image to look at.

[Prioritized Experience Replay](http://arxiv.org/abs/1511.05952) - drawing from the memory should be more likely if the memory is more shocking

[Deep Recurrent Q-Learning For Partially Observable MDPs](http://arxiv.org/pdf/1507.06527.pdf) - by using LSTM you can get rid of preprocessing done in DQN paper. 
"The recurrent net can better adapt at evaluation time if the quality of observations changes"

[A fast learning algorithm for deep belief nets](http://www.cs.toronto.edu/~hinton/absps/fastnc.pdf) - Training one layer at a time

[Reinforcement Learning and Automated Planning: A Survey](http://lpis.csd.auth.gr/publications/rlplan.pdf)

[Autoregressive Neural Networks](https://opus4.kobv.de/opus4-uni-passau/files/142/Dietz_Sebastian.pdf) - Neural Networks applied to Time Series.

[Deep Autoregressive Neural Networks](https://www.cs.toronto.edu/~amnih/papers/darn.pdf) - predicting future frames of an Atari Game.

[Reinforcement Learning: An introduction](https://webdocs.cs.ualberta.ca/~sutton/book/ebook/) - very thorough introduction to Reinforcement Learning.

[A survey of robot learning by demonstration](http://www.cs.cmu.edu/~mmv/papers/09ras-survey.pdf)
