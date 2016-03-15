import run

d = run.d
d['dqn.no_replay'] = True
d['visualize'] = 'q'
d['dqn.replay_start_size'] = 100
d['show_mood'] = run.Log
run.main(**d)
