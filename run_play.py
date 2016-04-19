import run
import network
import updates

d = run.d
d['game'] = 'space_invaders'
d['reshape'] = 'mean'
d['phi'] = 'phi2'
d['screen_size'] = 84
d['dqn.no_replay'] = True

d['visualize'] = 'ale'
d['dqn.replay_start_size'] = 100
d['dqn.log_frequency'] = 10
d['dqn.final_epsilon'] = 0.1
d['dqn.initial_epsilon'] = 0.1
d['weights_dir'] = 'dqn20/weights'
d['dqn.network'] = network.build_nature_with_pad
#d['dqn.updates'] = updates.deepmind_rmsprop
d['show_mood'] = run.Plot
run.main(**d)
