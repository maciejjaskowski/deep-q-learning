#def dqn_on_space_invaders_gpu(visualize=False, theano_verbose=False, initial_weights_file=None, initial_i_frame=0):
import q_learning as q
import ale_game as ag
import dqn
import theano
import theano.tensor as T
import lasagne
import numpy as np


def latest(dir='.'):
    if dir == None:
        return None, 0
    import os, re
    frames = [int(re.match(r"weights_([0-9]*).npz", file).groups()[0])
             for file in os.listdir(dir) if file.startswith("weights_")]
    if frames == None or len(frames) == 0:
        return None, 0
    else:
        return dir + '/weights_' + str(max(frames)) + '.npz', max(frames)


def main(**kargs):
    weights_file, _ = latest(kargs['weights_dir'])

    print("Continuing using weights from file: ", weights_file)

    if kargs['theano_verbose']:
        theano.config.compute_test_value = 'warn'
        theano.config.exception_verbosity = 'high'
        theano.config.optimizer = 'fast_compile'

    build_cnn = kargs['dqn.network']
    updates = kargs['dqn.updates']
    learning_rate = kargs['learning_rate']
    n_actions = 6

    initial = np.random.random((1, 4, 80, 80)).astype(theano.config.floatX)
    s0_var = theano.shared(initial)
    network = build_cnn(n_actions=n_actions, input_var=s0_var)
    with np.load(weights_file) as initial_weights:
        param_values = [initial_weights['arr_%d' % i] for i in range(len(initial_weights.files))]
        lasagne.layers.set_all_param_values(network, param_values)


    out = lasagne.layers.get_output(network)
    loss = -out[0][1] # shoot
    params = lasagne.layers.get_all_params(network, trainable=True)

    print("Compiling train_fn.")
    train_fn = theano.function([], outputs=[out, loss],
                                   updates=updates(loss, [s0_var], learning_rate))




    keep_as_img_fn = theano.function([], outputs=[], updates=[(s0_var, T.min(T.stack(T.ones_like(s0_var),
                                                                                     T.max(T.stack(T.zeros_like(s0_var), s0_var), axis=0)),
                                                                             axis=0))])

    from theano.tensor.shared_randomstreams import RandomStreams
    srng = RandomStreams(seed=234)


    prev = initial
    jitter = 8
    loss = 0
    n_report = 1000.0
    for i in range(100000):
        ox, oy = np.random.randint(-jitter, jitter+1, 2)
        s0_var.set_value(np.roll(np.roll(s0_var.get_value(), ox, -1), oy, -2)) # apply jitter shift
        loss = loss + train_fn()[0][0]
        s0_var.set_value(np.roll(np.roll(s0_var.get_value(), -ox, -1), -oy, -2))
        keep_as_img_fn()


        if np.any(np.isnan(s0_var.get_value())):
            break

        if i % n_report == 0:

            print(i)
            np.savez('dream/dream_{0:06d}.npz'.format(i), s0_var.get_value())
            print("diff: ", np.sum(s0_var.get_value() - prev))
            print("abs: ", np.sum(s0_var.get_value()))
            print("loss: ", loss / n_report)
            print()
            loss = 0

        prev = s0_var.get_value()




def vis():
    import os
    import ale_game as ag
    vis = ag.SpaceInvadersGameCombined2Visualizer()
    paths = sorted(['dream/' + f for f in os.listdir('dream')])

    for path in paths:
        with np.load(path) as screen:
            print(screen)
            vis.show(screen['arr_0'] * 256)
            raw_input("Press Enter to continue...")

#vis()

d = {
    'weights_dir': 'dqn16/weights',
    'theano_verbose': False,
    'dqn.network': dqn.build_nature_cnn,
    'dqn.updates': lasagne.updates.rmsprop,
    'learning_rate': 0.005
     }

if __name__ == "__main__":
    import sys
    import getopt
    optlist, args = getopt.getopt(sys.argv[1:], '', [
        'theano_verbose=',
        'weights_dir=',
        'dqn.network=',
        'dqn.updates=',
        'learning_rate='])

    for o, a in optlist:
        if o in ("--weights_dir",):
            d['weights_dir'] = a
        elif o in ("--theano_verbose",):
            d["theano_verbose"] = bool(a)
        elif o in ("--dqn.network",):
            if a == 'cnn':
                d["dqn.network"] = dqn.build_nature_cnn
            elif a == 'cnn_gpu':
                d["dqn.network"] = dqn.build_nature_cnn_gpu
            elif a == 'nips_cnn_gpu':
                d["dqn.network"] = dqn.build_nips_cnn_gpu
        elif o in ("--dqn.updates",):
            import updates
            if a == 'deepmind_rmsprop':
                d["dqn.updates"] = \
                    lambda loss, params: updates.deepmind_rmsprop(loss, params, learning_rate=.00025, rho=.95, epsilon=.1)
            elif a == 'rmsprop':
                d["dqn.updates"] = \
                    lambda loss, params: lasagne.updates.rmsprop(loss, params, learning_rate=.0002, rho=.95, epsilon=1e-6)
        elif o in ("--learning_rate",):
            d['learning_rate'] = float(a)
        else:
            assert False, "unhandled option"

    import pprint
    pp = pprint.PrettyPrinter(depth=2)
    print(optlist)
    print(args)
    print(sys.argv)
    print("")
    pp.pprint(d)

    main(**d)

