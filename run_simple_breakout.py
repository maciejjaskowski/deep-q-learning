import run
import network

d = run.d
d['game'] = 'simple_breakout'
d['replay'] = 'uniform'
d['visualize'] = 'q'
d['dqn.replay_start_size'] = 50000
d['show_mood'] = None
#d['final_exploration_action'] = 100000
d['dqn.network'] = network.build_simple_breakout
import updates
d['dqn.updates'] = lambda loss, params: updates.deepmind_rmsprop(loss, params, learning_rate=.00025, rho=.95, epsilon=.01)
#d['weights_dir'] = 'weights'
#d['algo'] = 'dqn'
#d['last_action_no'] = 2000000
#d['max_actions_per_game'] = 1000
d['repeat_action'] = 1
d['skip_n_frames_after_lol'] = 0
d['target_network_update_frequency'] = 10000
d['final_exploration_frame'] = 1000000
d['dqn.replay_memory_size'] = 400000
d['run_test_every_n'] = 50000
run.main(**d)
