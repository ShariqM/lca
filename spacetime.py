import matplotlib
matplotlib.rcParams.update({'figure.autolayout': True}) # Magical tight layout

import traceback
import math
import socket
import os
import scipy.io
import scipy.stats as stats
from datetime import datetime as dt
import time
import numpy as np
from numpy import reshape, zeros, ones
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from math import floor, ceil, sqrt
import pdb
from st_helpers import *
import sys
import random

dtype = theano.config.floatX
class SpaceTime():

    # Parameters
    #patch_dim = 144
    #neurons   = 200
    #timepoints = 7
    patch_dim = 64
    neurons   = 96
    timepoints = 5

    batch_size = 10
    time_batch_size = 64
    num_trials = 1000

    eta_init = 6.0
    eta_inc  = 100

    citers    = 60
    coeff_eta = 0.05
    lambdav   = 0.1

    data_name = 'IMAGES_DUCK_SHORT'
    profile = True
    show = False

    # Inferred parameters
    sz = int(np.sqrt(patch_dim))

    def __init__(self):
        self.images = scipy.io.loadmat('mat/%s.mat' % self.data_name)
        self.images = self.images[self.data_name]
        (self.imsize, imsize, self.num_images) = np.shape(self.images)
        self.patch_per_dim = int(np.floor(imsize / self.sz))
        plt.ion()

    def load_videos(self):
        imsize, sz, tbs = self.imsize, self.sz, self.time_batch_size
        VI = np.zeros((self.patch_dim, self.batch_size, self.time_batch_size))
        for x in range(self.batch_size):
            # Choose a random image less than time_batch_size images
            # away from the end
            imi = np.floor((self.num_images - tbs) * random.uniform(0, 1))
            r = (imsize-sz) * random.uniform(0, 1)
            c = (imsize-sz) * random.uniform(0, 1)
            img =  self.images[r:r+sz, c:c+sz, imi:imi+tbs]
            VI[:,x,:] = np.reshape(img, (self.patch_dim, tbs), 1)
        return VI

    def normalize_Phi(self, Phi, a):
        #Phi = np.einsum('pnt,nn->pnt', Phi, np.diag(np.average(np.sum(a ** 2, axis=2), axis=1)))
        for i in range(self.timepoints):
            Phi[:,:,i] = np.dot(Phi[:,:,i], np.diag(1/np.sqrt(np.sum(Phi[:,:,i]**2, axis = 0))))
        return Phi

    def get_eta(self, trial):
        eta = self.eta_init
        for i in range(10):
            if trial < self.eta_inc * (i+1):
                return eta/self.batch_size
            eta /= 2.0
        return eta/self.batch_size

    def sparsify(self, I, Phi, num_iterations=80):
        pass

    def error(self, VI, Phi, a):
        e = np.zeros((self.patch_dim, self.batch_size, self.time_batch_size))
        start = dt.now()
        for t in range(self.time_batch_size):
            I = VI[:,:,t]
            size = min(self.timepoints - 1, t)
            e[:,:,t] = I - np.einsum('pnt,nbt->pb', Phi[:,:,0:size+1], a[:,:,t::-1][:,:,0:size+1])
            e[:,:,t] = I - tendot('pnt,nbt->pb', Phi[:,:,0:size+1], a[:,:,t::-1][:,:,0:size+1])
        self.profile_print("Error Calc", start)
        return e

    def a_cot(self, Phi, e):
        'Correlation over time'
        start = dt.now()
        result = np.zeros((self.neurons, self.batch_size, self.time_batch_size))
        for t in range(self.time_batch_size):
            size = min(self.timepoints, self.time_batch_size - t)
            result[:,:,t] = np.einsum('pnt,pbt->nb', Phi[:,:,0:size], e[:,:,t:t+size])
        self.profile_print("dA Calc", start)
        return result

    def phi_cot(self, a, e):
        'Correlation over time'
        start = dt.now()
        result = np.zeros((self.patch_dim, self.neurons, self.timepoints))
        for tau in range(self.timepoints):
            for t in range(self.time_batch_size):
                if t - tau < 0:
                    break
                if t+tau >= self.time_batch_size:
                    break
                result[:,:,tau] = np.einsum('nb,pb->pn', a[:,:,t-tau], e[:,:,t+tau])
        self.profile_print("dPhi2 Calc", start)
        return result

    def profile_print(self, msg, start):
        if not self.profile:
            return
        diff = dt.now() - start
        print '%10s | E=%s' % (msg, diff)

    def sparse_cost(self, a):
        return (2 * a) / (1 + a ** 2)

    def get_activity(self, a):
        max_active = self.batch_size * self.time_batch_size * self.neurons
        ac = np.copy(a)
        #ac[np.abs(ac) > 0.01] = 1
        #ac[np.abs(ac) <= 0.00] = 0
        ac[np.abs(ac) > self.lambdav] = 1
        ac[np.abs(ac) <= self.lambdav] = 0
        return 100 * np.sum(ac)/max_active

    def train(self):
        Phi = np.random.randn(self.patch_dim, self.neurons, self.timepoints)
        if False:
            Phi[:,:,0] = np.dot(Phi[:,:,0], np.diag(1/np.sqrt(np.sum(Phi[:,:,0]**2, axis = 0))))
            for i in range(1, self.timepoints):
                Phi[:,:,i] = Phi[:,:,i-1] + 0.1 * np.random.randn(self.patch_dim, self.neurons)
                Phi[:,:,i] = np.dot(Phi[:,:,i], np.diag(1/np.sqrt(np.sum(Phi[:,:,i]**2, axis = 0))))
        else:
            for i in range(self.timepoints):
                Phi[:,:,i] = np.dot(Phi[:,:,i], np.diag(1/np.sqrt(np.sum(Phi[:,:,i]**2, axis = 0))))
        self.showbfs(Phi)

        for trial in range(self.num_trials):
            VI = self.load_videos()
            a = np.zeros((self.neurons, self.batch_size, self.time_batch_size))

            for c in range(self.citers):
                e = self.error(VI, Phi, a)
                if c == self.citers or c % 10 == 0:
                    print '\t%d) E=%.3f Activity=%.2f%%' % \
                        (c, np.sum(np.abs(e)), self.get_activity(a))
                da = self.a_cot(Phi, e) - self.lambdav * self.sparse_cost(a)
                a += self.coeff_eta * da

            dPhi = self.phi_cot(a, e)
            Phi += self.get_eta(trial) * dPhi
            e = self.error(VI, Phi, a)
            print '%d) E1=%.3f' % (trial, np.sum(np.abs(e)))
            Phi = self.normalize_Phi(Phi, a)
            e = self.error(VI, Phi, a)
            print '%d) E2=%.3f' % (trial, np.sum(np.abs(e)))
            self.showbfs(Phi)

    def showbfs(self, Phi):
        if not self.show:
            return
        L = self.patch_dim
        M = self.neurons
        sz = self.sz
        n = floor(sqrt(M)) # sz of one side of the grid of images
        m = ceil(M/n) # ceil for 1 extra
        buf = 1

        sz = self.sz

        for t in range(self.timepoints):
            arr = ones(shape=(buf + n * (sz + buf), buf + m * (sz + buf)))
            for k in range(M):
                i = (k % n)
                j = floor(k/n)

                def index(x):
                    return buf + x * (sz + buf)

                maxA=max(abs(Phi[:,k,t])) # RESCALE
                img = reshape(Phi[:,k,t], (sz, sz), order='C').transpose()/maxA
                arr[index(i):index(i)+sz, index(j):index(j)+sz] = img

            plt.imshow(arr, cmap = cm.binary, interpolation='none')
            plt.title('Phi @ t=%d' % t)
            plt.draw()
            plt.show()
            time.sleep(0.5)


st = SpaceTime()
st.train()
