
import sys
import os
from PIL import Image
import numpy as np
import ale_game as ag
import network
import theano.tensor as T
import theano
import lasagne

directory = sys.argv[1]
frame = int(sys.argv[2])

directory = "dqn16/record/"

weights_file = "dqn39/weights_8000100.npz"





s0_var = T.tensor4("s0", dtype=theano.config.floatX)
input_var = s0_var
n = network.build_nature_with_pad(6, input_var, 84)
out = lasagne.layers.get_output(n, s0_var)

outs = [lasagne.layers.get_output(layer, s0_var) for layer in lasagne.layers.get_all_layers(n)]

with np.load(weights_file) as initial_weights:
    param_values = [initial_weights['arr_%d' % i] for i in range(len(initial_weights.files))]
    lasagne.layers.set_all_param_values(n, param_values)

ff = theano.function([s0_var], outs)

import matplotlib.pyplot as plt
inputs = []
for frame in range(1200, 1400, 20):
    files = [os.path.join(directory, "%06d.png" % i) for i in range(frame, frame+16)]
    frames = [np.array(Image.open(f)) for f in files]
    gray_frames = [(np.dot(f, np.array([0.2126, 0.7152, 0.0722]))).astype(np.float32)  for f in frames]

    phi = ag.Phi(method="resize")
    inputs.append(np.stack(phi(gray_frames), axis=0).reshape(1, 4, 84, 84))



output = plt.imshow(ff(inputs[0])[1][0][0], cmap='Greys_r')
print(output)

