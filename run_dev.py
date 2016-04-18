import run
import network

d = run.d
d['game'] = 'space_invaders'
d['reshape'] = 'mean'
d['dqn.no_replay'] = True
d['visualize'] = 'q'
d['dqn.replay_start_size'] = 100
d['show_mood'] = run.Log
d['dqn.network'] = network.build_nature
d['weights_dir'] = None
#d['weights_dir'] = 'weights'
run.main(**d)
