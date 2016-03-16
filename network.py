from __future__ import print_function
from __future__ import division

import lasagne


def build_nature_dnn(n_actions, input_var):
    from lasagne.layers import dnn

    l_in = lasagne.layers.InputLayer(
        shape=(32, 4, 80, 80),
        input_var=input_var
    )

    l_conv1 = dnn.Conv2DDNNLayer(
        l_in,
        num_filters=32,
        filter_size=(8, 8),
        stride=(4, 4),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.HeUniform(),
        b=lasagne.init.Constant(.1)
    )

    l_conv2 = dnn.Conv2DDNNLayer(
        l_conv1,
        num_filters=64,
        filter_size=(4, 4),
        stride=(2, 2),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.HeUniform(),
        b=lasagne.init.Constant(.1)
    )

    l_conv3 = dnn.Conv2DDNNLayer(
        l_conv2,
        num_filters=64,
        filter_size=(3, 3),
        stride=(1, 1),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.HeUniform(),
        b=lasagne.init.Constant(.1)
    )

    l_hidden1 = lasagne.layers.DenseLayer(
        l_conv3,
        num_units=512,
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.HeUniform(),
        b=lasagne.init.Constant(.1)
    )

    l_out = lasagne.layers.DenseLayer(
        l_hidden1,
        num_units=n_actions,
        nonlinearity=None,
        W=lasagne.init.HeUniform(),
        b=lasagne.init.Constant(.1)
    )

    return l_out


def build_nature_with_pad(n_actions, input_var=None):
    network = lasagne.layers.InputLayer(shape=(None, 4, 80, 80),
                                        input_var=input_var)

    network = lasagne.layers.Conv2DLayer(
        network, num_filters=32, filter_size=(8, 8), stride=4, pad=2,
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform(),
        b=lasagne.init.Constant(.1))

    print(network.output_shape)

    network = lasagne.layers.Conv2DLayer(
        network, num_filters=64, filter_size=(4, 4), stride=2,
        nonlinearity=lasagne.nonlinearities.rectify,
        b=lasagne.init.Constant(.1))

    network = lasagne.layers.Conv2DLayer(
        network, num_filters=64, filter_size=(3, 3), stride=1,
        nonlinearity=lasagne.nonlinearities.rectify,
        b=lasagne.init.Constant(.1))

    network = lasagne.layers.DenseLayer(
        network,
        num_units=512,
        nonlinearity=lasagne.nonlinearities.rectify,
        b=lasagne.init.Constant(.1))

    network = lasagne.layers.DenseLayer(
        network,
        num_units=n_actions,
        b=lasagne.init.Constant(.1))

    return network


def build_nature(n_actions, input_var=None):
    network = lasagne.layers.InputLayer(shape=(None, 4, 80, 80),
                                        input_var=input_var)

    network = lasagne.layers.Conv2DLayer(
        network, num_filters=32, filter_size=(8, 8), stride=4,
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform(),
        b=lasagne.init.Constant(.1))

    print(network.output_shape)

    network = lasagne.layers.Conv2DLayer(
        network, num_filters=64, filter_size=(4, 4), stride=2,
        nonlinearity=lasagne.nonlinearities.rectify,
        b=lasagne.init.Constant(.1))

    network = lasagne.layers.Conv2DLayer(
        network, num_filters=64, filter_size=(3, 3), stride=1,
        nonlinearity=lasagne.nonlinearities.rectify,
        b=lasagne.init.Constant(.1))

    network = lasagne.layers.DenseLayer(
        network,
        num_units=512,
        nonlinearity=lasagne.nonlinearities.rectify,
        b=lasagne.init.Constant(.1))

    network = lasagne.layers.DenseLayer(
        network,
        num_units=n_actions,
        b=lasagne.init.Constant(.1))

    return network


def build_nips(n_actions, input_var):
    network = lasagne.layers.InputLayer(shape=(32, 4, 80, 80),
                                        input_var=input_var)

    network = lasagne.layers.Conv2DLayer(
        network, num_filters=16, filter_size=(8, 8), stride=4,
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform())

    network = lasagne.layers.Conv2DLayer(
        network, num_filters=32, filter_size=(4, 4), stride=2,
        nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers.DenseLayer(
        network,
        num_units=256,
        nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers.DenseLayer(
        network,
        num_units=n_actions,
        nonlinearity=None,
        W=lasagne.init.HeUniform(),
        b=lasagne.init.Constant(.1))

    return network


def build_nips_dnn(n_actions, input_var):
    from lasagne.layers import dnn

    network = lasagne.layers.InputLayer(shape=(32, 4, 80, 80),
                                        input_var=input_var)

    network = dnn.Conv2DDNNLayer(
        network, num_filters=16, filter_size=(8, 8), stride=4,
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform())

    network = dnn.Conv2DDNNLayer(
        network, num_filters=32, filter_size=(4, 4), stride=2,
        nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers.DenseLayer(
        network,
        num_units=256,
        nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers.DenseLayer(
        network,
        num_units=n_actions,
        nonlinearity=None,
        W=lasagne.init.HeUniform(),
        b=lasagne.init.Constant(.1))

    return network
