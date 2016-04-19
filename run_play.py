import run
import network

d = run.d
d['game'] = 'space_invaders'
d['reshape'] = 'mean'
d['phi'] = 'phi2'
d['screen_size'] = 84
d['dqn.no_replay'] = True
d['visualize'] = 'q'
d['dqn.network'] = network.build_nature_with_pad
d['dqn.replay_start_size'] = 100
d['dqn.log_frequency'] = 1
d['dqn.final_epsilon'] = 0.05
d['dqn.initial_epsilon'] = 0.05
d['weights_dir'] = 'dqn28/weights'
d['show_mood'] = run.Plot
d['run_test_every_n'] = 10000000000
run.main(**d)
