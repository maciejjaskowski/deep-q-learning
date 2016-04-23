import run
import network

d = run.d
d['game'] = 'simple_breakout'
d['replay'] = 'uniform'
d['visualize'] = None#'q'
d['replay_start_size'] = 50000
d['show_mood'] = None
#d['final_exploration_action'] = 100000
d['network'] = network.build_simple_breakout_W_caffe_normal
import updates
d['updates'] = 'deepmind_rmsprop'
d['weights_dir'] = 'weights'

#d['weights_dir'] = 'weights-sb-1'
#d['algo'] = 'dqn'
#d['last_action_no'] = 2000000
#d['max_actions_per_game'] = 1000
d['repeat_action'] = 1
d['skip_n_frames_after_lol'] = 0
d['target_network_update_frequency'] = 10000
d['final_exploration_frame'] = 100000
d['replay_memory_size'] = 400000
run.main(**d)
