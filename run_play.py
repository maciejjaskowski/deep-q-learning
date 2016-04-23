import run
import network
import updates

d = run.d
d['game'] = 'space_invaders'
d['reshape'] = 'mean'
d['phi'] = 'phi2'
d['screen_size'] = 84
d['no_replay'] = True

d['visualize'] = 'ale'
d['replay_start_size'] = 100
d['log_frequency'] = 10
d['final_epsilon'] = 0.1
d['initial_epsilon'] = 0.1
d['weights_dir'] = 'dqn20/weights'
d['network'] = network.build_nature_with_pad
#d['updates'] = updates.deepmind_rmsprop
d['show_mood'] = run.Plot
run.main(**d)
