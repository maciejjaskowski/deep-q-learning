import run
import network
import updates


d = run.d
d['game'] = 'space_invaders'
d['phi'] = '4'
d['network'] = 'nature_with_pad'
d['updates'] = 'deepmind_rmsprop'
d['run_test_every_n'] = 10000000000
d['weights_dir'] = 'dqn42/weights'
#d['dqn.replay_start_size'] = 100


run.main(**d)
