import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from showbfs import showbfs
import pdb

#Psi = np.load('dict/cul_spacetime_169_3.npy')
#Psi = np.load('dict/cul_spacetime_169_4_0.015.npy')
#Psi = np.load('dict/cul_spacetime_169_6_0.045.npy')
#Psi = np.load('dict/cul_spacetime_169_7_0.040.npy')
Psi = np.load('dict/cul_spacetime_169_8_0.270.npy')

cells, neurons, timepoints = Psi.shape

for c in range(cells):

    axis_height = 1.2 * np.max(np.abs(Psi[c,:,:]))
    plt.axis([0, timepoints - 1, -axis_height, axis_height])
    for n in range(neurons):
        plt.plot(range(timepoints), Psi[c,n,:], label='%d' % n)
    plt.title("Cell=%d" % c)
    plt.show()
