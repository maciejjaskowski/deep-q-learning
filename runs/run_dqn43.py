import run
import network
import updates


d = run.defaults
d['game_name'] = 'space_invaders'
d['phi_type'] = '4'
d['network_type'] = 'nature_with_pad'
d['updates_method'] = 'deepmind_rmsprop'
d['weights_dir'] = 'dqn43/weights'
d['n_test_epochs'] = 0
d['n_training_epochs'] = 500
#d['dqn.replay_start_size'] = 100

run.main(**d)
