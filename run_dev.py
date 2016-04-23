import run
import network

d = run.d
d['game'] = 'space_invaders'
d['reshape'] = 'mean'
#d['no_replay'] = True
d['visualize'] = None #'q'
d['replay_start_size'] = 100
d['show_mood'] = run.Log
d['network'] = network.build_nature
d['weights_dir'] = None
#d['weights_dir'] = 'weights'
run.main(**d)
