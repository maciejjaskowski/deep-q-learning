import run

d = run.d
d['dqn.no_replay'] = True
d['visualize'] = 'q'
d['dqn.replay_start_size'] = 100
d['dqn.log_frequency'] = 10
d['dqn.final_epsilon'] = 0.1
d['dqn.initial_epsilon'] = 0.1
d['weights_dir'] = 'weights'
d['show_mood'] = run.Plot
run.main(**d)
