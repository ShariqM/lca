import matplotlib
import scipy.io
import time
import numpy as np
from numpy import reshape, zeros, ones
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from math import floor, ceil, sqrt
import pprint
import argparse


imsz = 192
sz = 12
frames = sz * 200

data = np.zeros((imsz, imsz, frames))
t = 0
for iters in range(frames/12):
    for i in range(sz):
        frame = np.zeros((imsz,imsz))
        frame[:,[x for x in range(i, imsz, sz)]] = 1
        frame[:,[x+1 for x in range(i, imsz-1, sz)]] = -1
        data[:,:,t] = frame
        t = t + 1
        #plt.imshow(data ,norm=matplotlib.colors.Normalize(-1,1,True), cmap = cm.binary, interpolation='none')

        #plt.imshow(frame ,norm=matplotlib.colors.Normalize(-1,1,True), cmap = cm.binary)
        #plt.show()

print data.shape
plt.imshow(data[:,:,13] ,norm=matplotlib.colors.Normalize(-1,1,True), cmap = cm.binary)
plt.show()
scipy.io.savemat('mat/IMAGES_MOVE_RIGHT.mat', {'IMAGES_MOVE_RIGHT': data})


test = False
if test:
    oname = 'IMAGES_DUCK_SHORT'
    OIMAGES = scipy.io.loadmat('mat/%s.mat' % oname)[oname]

    plt.imshow(OIMAGES[:,:,0],norm=matplotlib.colors.Normalize(-1,1,True), cmap = cm.binary)
    plt.show()

    OIMAGES[:,[x for x in range(i, imsz, sz)], 0] = -1
    plt.imshow(OIMAGES[:,:,0],norm=matplotlib.colors.Normalize(-1,1,True), cmap = cm.binary)
    plt.show()
