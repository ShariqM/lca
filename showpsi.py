import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from showbfs import showbfs
import pdb

Psi = np.load('dict/cul_spacetime.npy')

cells, neurons, timepoints = Psi.shape
pdb.set_trace()

for c in range(cells):

    axis_height = np.max(np.abs(Psi[c,:,:]))
    plt.axis([0, timepoints, -axis_height, axis_height])
    for n in range(neurons):
        plt.plot(range(timepoints), Psi[c,n,:], label='%d' % n)
    plt.title("Cell=%d" % c)
    plt.show()
