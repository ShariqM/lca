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
import pdb

orig_show = True
smooth_show = False

#oname = 'IMAGES_DUCK_LONG'
#sname = 'IMAGES_DUCK_LONG_SMOOTH_0.7'

#oname = 'IMAGES_DUCK_LONG'
#sname = 'IMAGES_DUCK_LONG_FAKE_SMOOTH_0.7'

#oname = 'IMAGES_PATCH_DUCK'
#oname = 'IMAGES_DUCK_SHORT'
#oname = 'IMAGES_EDGE_RIGHT_DUCK'
oname = 'IMAGES_EDGE_DUCK'

oname = 'IMAGES_PHI_INTERP'
oname = 'IMAGES_PHI_463_INTERP'

#oname = 'IMAGES_EDGE_DUCK_2'
#oname = 'IMAGES_DUCK'
#oname = 'IMAGES_MOVE_RIGHT'
#sname = 'IMAGES_DUCK_SMOOTH_0.7'

#oname = 'IMAGES_DUCK_SHORT'
#sname = 'IMAGES_DUCK_SHORT_SMOOTH_0.7'

sleep = 0.00 # seconds to sleep between frames
interval = 1 # Show every *interval* frame


if smooth_show:
    SIMAGES = scipy.io.loadmat('mat/%s.mat' % sname)[sname]
    shape = SIMAGES.shape

if orig_show:
    if oname == 'IMAGES_DUCK_LONG':
        # Load data, have to use h5py because had to use v7.3 because .mat is so big.
        f = h5py.File('mat/%s.mat' % oname, 'r',)
        OIMAGES = np.array(f.get(oname))
        OIMAGES = np.swapaxes(OIMAGES, 0, 2) # v7.3 reorders for some reason, or h5?
    else:
        OIMAGES = scipy.io.loadmat('mat/%s.mat' % oname)[oname]

    shape = OIMAGES.shape

print 'Shape:', shape

plt.ion()
#start, stop = 70, 82
start, stop = 0, shape[2]
plt.figure(figsize=(8,8))
for i in range(start, stop, interval):
    if orig_show:
        #plt.subplot(211)
        r = 144
        c = 144
        #pdb.set_trace()
        #plt.imshow(OIMAGES[r:r+12,c:c+12,i], norm=matplotlib.colors.Normalize(-1,1,True), cmap = cm.binary, interpolation='none')
        plt.imshow(OIMAGES[r:r+12,c:c+12,i], norm=matplotlib.colors.Normalize(-0.25,0.25,True), cmap = cm.binary, interpolation='none')
        #plt.imshow(OIMAGES[24:36,24:36,i], norm=matplotlib.colors.Normalize(-1,1,True), cmap = cm.binary)
        #plt.imshow(OIMAGES[:,:,i], norm=matplotlib.colors.Normalize(-1,1,True), cmap = cm.binary, interpolation='none')
        plt.title('Original %d Var=%.4f' % (i, OIMAGES[:,:,i].var().mean()))
        time.sleep(0.2)

    if smooth_show:
        plt.subplot(212)
        plt.imshow(SIMAGES[:,:,i-150], norm=matplotlib.colors.Normalize(-1,1,True), cmap = cm.binary)
        plt.title('Smooth %d Var=%.4f' % (i, SIMAGES[:,:,i].var().mean()))

    plt.draw()
    plt.show()
    plt.clf()
    time.sleep(sleep)
