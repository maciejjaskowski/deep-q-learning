""" Utility to plot the first layer of convolutions learned by
the Deep q-network.
(Assumes dnn convolutions)
Usage:
plot_filters.py PICKLED_NN_FILE
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
