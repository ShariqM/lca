import matplotlib
import scipy.io
import time
import numpy as np
from numpy import reshape, zeros, ones
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from math import floor, ceil, sqrt
import pprint
import h5py
import argparse

oname = 'IMAGES_DUCK_LONG'
sname = 'IMAGES_DUCK_LONG_SMOOTH_0.7'
oname = 'IMAGES_DUCK_SHORT'
sname = 'IMAGES_DUCK_SHORT_SMOOTH_0.7'
#name = 'IMAGES_DUCK_SHORT_SMOOTH'

sleep = 0.05 # seconds

if oname == 'IMAGES_DUCK_LONG':
    # Load data, have to use h5py because had to use v7.3 because .mat is so big.
    f = h5py.File('mat/%s.mat' % oname, 'r',)
    OIMAGES = np.array(f.get(oname))
    OIMAGES = np.swapaxes(OIMAGES, 0, 2) # v7.3 reorders for some reason, or h5?
else:
    OIMAGES = scipy.io.loadmat('mat/%s.mat' % oname)[oname]

SIMAGES = scipy.io.loadmat('mat/%s.mat' % sname)[sname]


print 'Shape:', SIMAGES.shape
plt.ion()
for i in range(150, SIMAGES.shape[2]):
    plt.subplot(211)
    plt.imshow(OIMAGES[:,:,i], norm=matplotlib.colors.Normalize(-1,1,True), cmap = cm.binary)
    plt.title('Original %d Var=%.4f' % (i, OIMAGES[:,:,i].var().mean()))

    plt.subplot(212)
    plt.imshow(SIMAGES[:,:,i], norm=matplotlib.colors.Normalize(-1,1,True), cmap = cm.binary)
    plt.title('Smooth %d Var=%.4f' % (i, SIMAGES[:,:,i].var().mean()))

    plt.draw()
    plt.show()
    plt.clf()
    time.sleep(sleep)
