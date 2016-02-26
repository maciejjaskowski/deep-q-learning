THEANO_FLAGS='floatX=float32' python ex1.py --visualize=q --dqn.no_replay --dqn.log_frequency=10 --dqn.final_epsilon=0.1 --dqn.initial_epsilon=0.1 --weights_dir="dqn2/weights" --show_mood
