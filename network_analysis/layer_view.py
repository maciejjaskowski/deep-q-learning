""" Utility to plot the indicated layer of convolutions learned by
the Deep q-network, with the first layer corresponding to 0.
(Assumes dnn convolutions)
Usage:
python layer_view.py weight.npz layer_integer
"""

import sys
import matplotlib.pyplot as plt
import numpy

net_file = numpy.load(sys.argv[1])
array_index = (int(sys.argv[2])*2)
w = net_file['arr_'+str(array_index)]

count = 1
for c in range(w.shape[1]): # channels/time-steps
    for f in range(w.shape[0]): # filters
        plt.subplot(w.shape[1], w.shape[0], count)
        img = w[f, c, :, :]
        plt.imshow(img, vmin=img.min(), vmax=img.max(),
                   interpolation='none', cmap='gray')
        plt.xticks(())
        plt.yticks(())
        count += 1
plt.show()
