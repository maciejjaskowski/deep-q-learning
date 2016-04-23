import run
import network

d = run.d.copy()
d['game'] = 'space_invaders'
d['visualize'] = None
d['replay_start_size'] = 50000
d['show_mood'] = None
d['network'] = 'nature_with_pad_he'
d['updates'] = 'deepmind_rmsprop'

d['repeat_action'] = 4
d['skip_n_frames_after_lol'] = 30
d['target_network_update_frequency'] = 10000
d['final_exploration_frame'] = 1000000
d['replay_memory_size'] = 400000
run.main(**d)
