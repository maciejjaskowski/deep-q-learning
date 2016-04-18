import run
import network

d = run.d
d['game'] = 'simple_breakout'
d['replay'] = 'uniform'
d['visualize'] = None #'q'
d['dqn.replay_start_size'] = 5000
d['show_mood'] = None
#d['final_exploration_action'] = 100000
d['dqn.network'] = network.build_simple_breakout
#d['weights_dir'] = 'weights'
#d['algo'] = 'dqn'
#d['last_action_no'] = 2000000
#d['max_actions_per_game'] = 1000
d['repeat_action'] = 1
d['skip_n_frames_after_lol'] = 0
d['target_network_update_frequency'] = 5000
run.main(**d)
