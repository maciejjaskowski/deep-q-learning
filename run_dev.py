import run
import network

d = run.d
d['dqn.no_replay'] = True
d['visualize'] = 'q'
d['dqn.replay_start_size'] = 100
d['show_mood'] = run.Log
d['dqn.network'] = network.build_nature_with_pad3
run.main(**d)
