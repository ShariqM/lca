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
    profile = True

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

    def profile_print(self, msg, start):
        if not self.profile:
            return
        diff = dt.now() - start
        print '%10s | E=%s' % (msg, diff)

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
                start = dt.now()
                for t in range(self.time_batch_size):
                    I = VI[:,:,t]
                    e[:,:,t] = I - self.convolve(t, Phi, a)
                self.profile_print("E Calc", start)

                print '%d) E=%.3f' % (c, np.sum(np.abs(e)))

                start = dt.now()
                da = self.cot(Phi, e)
                self.profile_print("dA Calc", start)

                a += self.coeff_eta * da

            dPhi = 0
            Phi += self.get_eta(trial) * dPhi
            Phi = self.normalize_Phi(Phi)

st = SpaceTime()
st.train()

'''
0) E=7426.219
1) E=5177.111
2) E=4308.444
3) E=3767.701
4) E=3382.294
5) E=3087.779
6) E=2853.028
7) E=2660.199
8) E=2497.302
9) E=2356.864
10) E=2234.185
11) E=2125.351
12) E=2027.599
13) E=1939.026
14) E=1858.183
15) E=1783.910
16) E=1715.262
17) E=1651.470
18) E=1592.055
19) E=1536.475
20) E=1484.344
21) E=1435.355
22) E=1389.266
23) E=1345.834
24) E=1304.782
25) E=1265.887
26) E=1229.005
27) E=1193.985
28) E=1160.674
29) E=1128.928
30) E=1098.651
31) E=1069.770
32) E=1042.149
33) E=1015.716
34) E=990.441
35) E=966.263
36) E=943.056
37) E=920.765
38) E=899.332
39) E=878.731
'''
