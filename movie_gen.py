import matplotlib
import scipy.io
import scipy.stats as stats
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
        self.frames = self.sz * 100
        self.gen_type=gen_type

        self.data = np.zeros((self.imsz, self.imsz, self.frames))
        print self.data.shape

        self.show_images = False

    def save(self, name):
        scipy.io.savemat('mat/%s.mat' % name, {name: self.data})

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

        self.save('IMAGES_MOVE_RIGHT')

    def gauss(self,mean, point):
        return stats.norm(loc=mean,scale=0.5).pdf(point)

    def interp_right(self):
        raise Exception("Method not setup")
        t = 0
        for iters in range(self.frames/self.sz):
            for i in range(self.sz):
                frame = np.zeros((self.imsz,self.imsz))
                for c in range(self.imsz):
                    frame[:, c] = self.gauss(i, c % self.sz)
                self.data[:,:,t] = frame
                t = t + 1

                if self.show_images:
                    #plt.imshow(frame ,norm=matplotlib.colors.Normalize(-1,1,True), cmap = cm.binary)
                    plt.imshow(frame[0:12,0:12] ,norm=matplotlib.colors.Normalize(-1,1,True), cmap = cm.binary, interpolation='none')
                    plt.show()

        self.save('IMAGES_INTERP_RIGHT')

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

        self.save('IMAGES_BOUNCE')


    def duck_edge(self):
        oname = 'IMAGES_DUCK_SHORT'
        OIMAGES = scipy.io.loadmat('mat/%s.mat' % oname)[oname]

        window = 1
        r_idx = 12
        c_idx = 14
        start, stop = 0, 15
        t = 0
        plt.ion()
        for iters in range(self.frames / (2 * (stop - start))):
            for i in range(start, stop):
                for r in range(self.imsz/self.sz):
                    for c in range(self.imsz/self.sz):
                        self.data[r*self.sz:(r+1)*self.sz,c*self.sz:(c+1)*self.sz:,t] = OIMAGES[self.sz*(r_idx-1):self.sz*r_idx, self.sz*(c_idx-1):self.sz*c_idx,i]
                if self.show_images:
                    plt.imshow(OIMAGES[self.sz*(r_idx-1):self.sz*(r_idx+window-1), self.sz*(c_idx-1):self.sz*(c_idx+window-1),i], norm=matplotlib.colors.Normalize(-1,1,True), cmap = cm.binary)
                    plt.title("T=%d" % i)
                    plt.draw()
                    plt.show()

                t = t + 1

            for j in range(stop-2, start, -1):
                for r in range(self.imsz/self.sz):
                    for c in range(self.imsz/self.sz):
                        self.data[r*self.sz:(r+1)*self.sz,c*self.sz:(c+1)*self.sz:,t] = OIMAGES[self.sz*(r_idx-1):self.sz*r_idx, self.sz*(c_idx-1):self.sz*c_idx,j]
                if self.show_images:
                    plt.imshow(OIMAGES[self.sz*(r_idx-1):self.sz*r_idx, self.sz*(c_idx-1):self.sz*c_idx,j], norm=matplotlib.colors.Normalize(-1,1,True), cmap = cm.binary)
                    plt.draw()
                    plt.show()
                t = t + 1

        self.save('IMAGES_EDGE_DUCK')

    def duck_edge_right(self):
        oname = 'IMAGES_DUCK_SHORT'
        OIMAGES = scipy.io.loadmat('mat/%s.mat' % oname)[oname]

        window = 1
        r_idx = 12
        c_idx = 14
        start, stop = 0, 15
        t = 0
        plt.ion()
        for iters in range(self.frames / (stop - start)):
            for i in range(start, stop):
                for r in range(self.imsz/self.sz):
                    for c in range(self.imsz/self.sz):
                        self.data[r*self.sz:(r+1)*self.sz,c*self.sz:(c+1)*self.sz:,t] = OIMAGES[self.sz*(r_idx-1):self.sz*r_idx, self.sz*(c_idx-1):self.sz*c_idx,i]
                if self.show_images:
                    plt.imshow(OIMAGES[self.sz*(r_idx-1):self.sz*(r_idx+window-1), self.sz*(c_idx-1):self.sz*(c_idx+window-1),i], norm=matplotlib.colors.Normalize(-1,1,True), cmap = cm.binary)
                    plt.title("T=%d" % i)
                    plt.draw()
                    plt.show()

                t = t + 1
        self.save('IMAGES_EDGE_RIGHT_DUCK')

    def duck_patch(self):
        oname = 'IMAGES_DUCK_SHORT'
        OIMAGES = scipy.io.loadmat('mat/%s.mat' % oname)[oname]
        r_idx = 4
        c_idx = 9
        start, stop = 195, 220
        start, stop = 0, 450
        t = 0
        plt.ion()
        for iters in range(self.frames / (2 * (stop - start))):
            for i in range(start, stop):
                for r in range(self.imsz/self.sz):
                    for c in range(self.imsz/self.sz):
                        self.data[r*self.sz:(r+1)*self.sz,c*self.sz:(c+1)*self.sz:,t] = OIMAGES[self.sz*(r_idx-1):self.sz*r_idx, self.sz*(c_idx-1):self.sz*c_idx,i]
                if self.show_images:
                    plt.imshow(OIMAGES[self.sz*(r_idx-1):self.sz*r_idx, self.sz*(c_idx-1):self.sz*c_idx,i], norm=matplotlib.colors.Normalize(-1,1,True), cmap = cm.binary)
                    plt.draw()
                    plt.show()
                    t = t + 1

            for j in range(stop-2, start, -1):
                for r in range(self.imsz/self.sz):
                    for c in range(self.imsz/self.sz):
                        self.data[r*self.sz:(r+1)*self.sz,c*self.sz:(c+1)*self.sz:,t] = OIMAGES[self.sz*(r_idx-1):self.sz*r_idx, self.sz*(c_idx-1):self.sz*c_idx,j]
                if self.show_images:
                    plt.imshow(OIMAGES[self.sz*(r_idx-1):self.sz*r_idx, self.sz*(c_idx-1):self.sz*c_idx,j], norm=matplotlib.colors.Normalize(-1,1,True), cmap = cm.binary)
                    plt.draw()
                    plt.show()
                t = t + 1

        self.save('IMAGES_PATCH_DUCK')

    def run(self):
        if self.gen_type == 1:
            self.movie_right()
        elif self.gen_type == 2:
            self.interp_right()
        elif self.gen_type == 3:
            self.movie_bounce()
        elif self.gen_type == 4:
            self.duck_edge()
        elif self.gen_type == 5:
            self.duck_edge_right()
        else:
            self.duck_patch()

mg = MovieGen(4)
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
