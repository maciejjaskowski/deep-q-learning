import run
import network

d = run.defaults
d['game_name'] = 'simple_breakout'
#d['replay'] = 'uniform'
d['visualize'] = None#'q'
d['replay_start_size'] = 50000
d['show_mood'] = None
d['network_type'] = network.build_simple_breakout_W_caffe_normal
d['updates_method'] = 'deepmind_rmsprop'
d['weights_dir'] = 'weights_breakout'

d['repeat_action'] = 1
d['skip_n_frames_after_lol'] = 0
d['target_network_update_frequency'] = 10000
d['final_exploration_frame'] = 100000
d['replay_memory_size'] = 400000
import pprint
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(d)
run.main(**d)
