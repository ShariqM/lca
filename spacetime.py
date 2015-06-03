import matplotlib
matplotlib.rcParams.update({'figure.autolayout': True}) # Magical tight layout

import traceback
import math
import socket
import os
import scipy.io
import scipy.stats as stats
from datetime import datetime
import time
import numpy as np
from numpy import reshape, zeros, ones
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from math import floor, ceil, sqrt
import pdb
from helpers import *
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

    eta_init = 0.6
    eta_inc  = 100

    data_name = 'IMAGES_DUCK_SHORT'

    # Inferred parameters
    sz = int(np.sqrt(patch_dim))

    def __init__(self):
        self.images = scipy.io.loadmat('mat/%s.mat' % self.data_name)
        self.images = self.images[self.data_name]
        (self.imsize, imsize, self.num_images) = np.shape(self.images)
        self.patch_per_dim = int(np.floor(imsize / self.sz))

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

    def normalize_Phi(self, Phi):
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

    def convolve(self, t, Phi, a):
        result = np.zeros((self.patch_dim, self.batch_size))
        for b in range(self.batch_size):
            for i in range(self.neurons):
                for tau in range(self.timepoints):
                    if t - tau < 0:
                        break
                    result[:,b] += a[i,b,t - tau] * Phi[:, i, tau]
        return result

    def cot(self, Phi, e):
        'Correlation over time'
        result = np.zeros((self.neurons, self.batch_size, self.time_batch_size))
        for b in range(self.batch_size):
            for i in range(self.neurons):
                for t in range(self.time_batch_size):
                    for tau in range(self.timepoints):
                        if t+tau >= self.time_batch_size:
                            continue
                        prod = np.dot(Phi[:,i,tau], e[:,b,t+tau])
                        result[i, b, t] += prod
                        if np.abs(prod) > 20.0:
                            print 'prod', prod
                            pdb.set_trace()
        return result

    def train(self):
        Phi = np.random.randn(self.patch_dim, self.neurons, self.timepoints)
        for i in range(self.timepoints):
            Phi[:,:,i] = np.dot(Phi[:,:,i], np.diag(1/np.sqrt(np.sum(Phi[:,:,i]**2, axis = 0))))
        self.coeff_eta = 0.05
        self.citers = 40

        for trial in range(self.num_trials):
            VI = self.load_videos()
            a = np.zeros((self.neurons, self.batch_size, self.time_batch_size))

            e = np.zeros((self.patch_dim, self.batch_size, self.time_batch_size))

            for c in range(self.citers):
                for t in range(self.time_batch_size):
                    I = VI[:,:,t]
                    e[:,:,t] = I - self.convolve(t, Phi, a)
                print '%d) E=%.3f' % (c, np.sum(np.abs(e)))
                da = self.cot(Phi, e)
                a += self.coeff_eta * da

            dPhi = 0
            Phi += self.get_eta(trial) * dPhi
            Phi = self.normalize_Phi(Phi)

st = SpaceTime()
st.train()
