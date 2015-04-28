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


class MovieGen():


    def __init__(self, gen_type=1):
        self.imsz = 192
        self.sz = 12
        self.frames = self.sz * 200
        self.gen_type=gen_type

        self.data = np.zeros((self.imsz, self.imsz, self.frames))
        print self.data.shape

        self.show_images = False

    def movie_right(self):
        t = 0
        for iters in range(self.frames/self.sz):
            for i in range(self.sz):
                frame = np.zeros((self.imsz,self.imsz))
                frame[:,[x for x in range(i, self.imsz, self.sz)]] = 1
                frame[:,[x+1 for x in range(i, self.imsz-1, self.sz)]] = -1
                self.data[:,:,t] = frame
                t = t + 1

                if self.show_images:
                    plt.imshow(frame ,norm=matplotlib.colors.Normalize(-1,1,True), cmap = cm.binary)
                    plt.show()
        scipy.io.savemat('mat/IMAGES_MOVE_RIGHT.mat', {'IMAGES_MOVE_RIGHT': self.data})

    def movie_bounce(self):
        t = 0
        for iters in range(self.frames / (2 * self.sz)):
            for i in range(self.sz):
                frame = np.zeros((self.imsz,self.imsz))
                frame[:,[x for x in range(i, self.imsz, self.sz)]] = 1
                frame[:,[x+1 for x in range(i, self.imsz-1, self.sz)]] = -1
                self.data[:,:,t] = frame
                t = t + 1

                if self.show_images:
                    plt.imshow(frame ,norm=matplotlib.colors.Normalize(-1,1,True), cmap = cm.binary)
                    plt.show()

            for i in range(1, self.sz-1):
                frame = np.zeros((self.imsz,self.imsz))
                frame[:,[x for x in range(self.sz-i-1, self.imsz, self.sz)]] = 1
                frame[:,[x+1 for x in range(self.sz-i-1, self.imsz-1, self.sz)]] = -1
                self.data[:,:,t] = frame
                t = t + 1

                if self.show_images:
                    plt.imshow(frame ,norm=matplotlib.colors.Normalize(-1,1,True), cmap = cm.binary)
                    plt.show()

        scipy.io.savemat('mat/IMAGES_BOUNCE.mat', {'IMAGES_BOUNCE': self.data})

    def run(self):
        if self.gen_type == 1:
            self.movie_right()
        else:
            self.movie_bounce()

mg = MovieGen(2)
mg.run()

#plt.imshow(data[:,:,13] ,norm=matplotlib.colors.Normalize(-1,1,True), cmap = cm.binary)
#plt.show()

#test = False
#if test:
    #oname = 'IMAGES_DUCK_SHORT'
    #OIMAGES = scipy.io.loadmat('mat/%s.mat' % oname)[oname]
#
    #plt.imshow(OIMAGES[:,:,0],norm=matplotlib.colors.Normalize(-1,1,True), cmap = cm.binary)
    #plt.show()
#
    #OIMAGES[:,[x for x in range(i, imsz, sz)], 0] = -1
    #plt.imshow(OIMAGES[:,:,0],norm=matplotlib.colors.Normalize(-1,1,True), cmap = cm.binary)
    #plt.show()
