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

    def error(self, VI, Phi, a):
        e = np.zeros((self.patch_dim, self.batch_size, self.time_batch_size))
        start = dt.now()
        for t in range(self.time_batch_size):
            I = VI[:,:,t]
            size = min(self.timepoints - 1, t)
            e[:,:,t] = I - np.einsum('pnt,nbt->pb', Phi[:,:,0:size+1], a[:,:,t::-1][:,:,0:size+1])
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
        x = np.zeros((self.patch_dim, self.neurons, self.timepoints))

        for b in range(self.batch_size):
            for p in range(self.patch_dim):
                for n in range(self.neurons):
                    for tau in range(self.timepoints):
                        for t in range(self.time_batch_size):
                            if t - tau < 0:
                                break
                            if t+tau >= self.time_batch_size:
                                break
                            x[p,n,tau] = a[n,b,t-tau] * e[p,b,t+tau]
        self.profile_print("dPhi Calc", start)

        start = dt.now()
        result = np.zeros((self.patch_dim, self.neurons, self.timepoints))
        for tau in range(self.timepoints):
            for t in range(self.time_batch_size):
                if t - tau < 0:
                    break
                if t+tau >= self.time_batch_size:
                    break
                result[:,:,tau] = np.einsum('nb,pb->pn', a[:,:,t-tau], e[:,:,t+tau])

        #for tau in range(self.timepoints):
            #end = self.time_batch_size - tau
            #end2 = self.time_batch_size
            #result[:,:,tau] = np.einsum('nbt,pbt->pn', a[:,:,0:end], e[:,:,tau:end2])
        #if not np.allclose(result, x):
            #pdb.set_trace()
        self.profile_print("dPhi2 Calc", start)

        #for t in range(self.timepoints):
            #size = self.time_batch_size
            #result[:,:,t] = np.einsum('nbt,pbt->pn', a[:,:,t::-1][:,:,0:size+1],
                                      #e[:,:,t:t+size+1])
#
        #for t in range(self.time_batch_size):
            #size = min(self.timepoints - 1, self.time_batch_size - t - 1)
            #size = min(size, t)
            #print 't=%d, size=%d' % (t, size)
            #try:
                #result[:,:,t] = np.einsum('nbt,pbt->pn', a[:,:,t::-1][:,:,0:size+1],
                                          #e[:,:,t:t+size+1])
            #except Exception as ex:
                #pdb.set_trace()
        #self.profile_print("dPhi Calc", start)
        return result

    def profile_print(self, msg, start):
        if not self.profile:
            return
        diff = dt.now() - start
        print '%10s | E=%s' % (msg, diff)

    def train(self):
        load = False
        if not load:
            Phi = np.random.randn(self.patch_dim, self.neurons, self.timepoints)
            for i in range(self.timepoints):
                Phi[:,:,i] = np.dot(Phi[:,:,i], np.diag(1/np.sqrt(np.sum(Phi[:,:,i]**2, axis = 0))))
            np.save('tmp_phi', Phi)
        else:
            Phi = np.load('tmp_phi.npy')

        self.coeff_eta = 0.05
        self.citers = 40

        for trial in range(self.num_trials):
            VI = self.load_videos()
            a = np.zeros((self.neurons, self.batch_size, self.time_batch_size))

            if not load:
                for c in range(self.citers):
                    e = self.error(VI, Phi, a)
                    print '%d) E=%.3f' % (c, np.sum(np.abs(e)))
                    da = self.a_cot(Phi, e)
                    a += self.coeff_eta * da
                np.save('tmp_a', a)
            else:
                a = np.load('tmp_a.npy')
                e = self.error(VI, Phi, a)
                print 'E=%.3f' % (np.sum(np.abs(e)))
            dPhi = self.phi_cot(a, e)
            #pdb.set_trace()
            Phi += self.get_eta(trial) * dPhi
            e = self.error(VI, Phi, a)
            print 'E2=%.3f' % (np.sum(np.abs(e)))
            Phi = self.normalize_Phi(Phi)

st = SpaceTime()
st.train()
