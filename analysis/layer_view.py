""" Utility to plot the indicated layer of convolutions learned by
the Deep q-network, with the first layer corresponding to 0.
(Assumes dnn convolutions)
Usage:
python layer_view.py weight.npz layer_integer
"""

import sys
import matplotlib.pyplot as plt
import numpy
import matplotlib.animation as animation

net_file = numpy.load(sys.argv[1])
array_index = (int(sys.argv[2])*2)
w = net_file['arr_'+str(array_index)]

fig, _ = plt.subplots()
#for c in range(w.shape[1]): # channels/time-steps

[w / numpy.linalg.norm(v) for v in w]
#w = w / numpy.linalg.norm(w, axis=1)
#w = w / numpy.lina

ims = []
def init():
    for f in range(w.shape[0]): # filters
        sqrt = numpy.ceil(numpy.sqrt(w.shape[0]))
        print(sqrt)
        plt.subplot(sqrt, sqrt, f+1)
        img = w[f, 0, :, :]
        ims.append(plt.imshow(img, vmin=img.min(), vmax=img.max(),
                       interpolation='none', cmap='gray'))
        plt.xticks(())
        plt.yticks(())
    return ims

def run(c):
    for f in range(w.shape[0]): # filters
        ims[f].set_array(w[f, c%w.shape[1], :, :])
    return ims

def inf():
    cnt = 0
    while True:
        cnt -= 1
        if cnt < 0:
            cnt += w.shape[1]
        yield cnt

#run()
ani = animation.FuncAnimation(fig, run, inf, blit=False, interval=500,
         repeat=True, init_func=init)
plt.show()
