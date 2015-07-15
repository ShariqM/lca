import matplotlib
matplotlib.rcParams.update({'figure.autolayout': True}) # Magical tight layout

import socket
if 'eweiss' in socket.gethostname():
    matplotlib.use('Agg') # Don't crash because $Display is not set correctly on the cluster

import traceback
from math import log
import math
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
    patch_dim  = 144
    neurons    = 200
    cells      = 200
    timepoints = 2

    load_phi   = True
    save_phi   = True
    save_often = 5
    batch_size = 10
    time_batch_size = 128
    num_trials = 1000

    eta_init = 0.15
    eta_inc  = 32

    citers    = 70
    coeff_eta = 0.1
    norm_Phi  = 0.10
    lambdav   = 1.00

    data_name = 'IMAGES_DUCK_SHORT'
    phi_name = 'Phi_169_45.0'
    profile = False
    visualizer = False
    show = True

    # Inferred parameters
    sz = int(np.sqrt(patch_dim))
    graphics_initialized = False # Don't change

    def __init__(self):
        self.images = scipy.io.loadmat('mat/%s.mat' % self.data_name)
        self.images = self.images[self.data_name]
        (self.imsize, imsize, self.num_images) = np.shape(self.images)
        self.patch_per_dim = int(np.floor(imsize / self.sz))

        if self.phi_name != '':
            Phi = scipy.io.loadmat('dict/%s' % self.phi_name)
            self.Phi = Phi['Phi']
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

    def normalize_Phi(self, Phi):
        start = dt.now()
        for i in range(self.neurons):
            Phi[:,i,:] *= self.norm_Phi/np.linalg.norm(Phi[:,i,:])
        self.profile_print('normPhi', start)
        return Phi

    def get_eta(self, trial):
        eta = self.eta_init
        for i in range(10):
            if trial < self.eta_inc * (i+1):
                return eta/self.batch_size
            eta /= 2.0
        return eta/self.batch_size

    def get_reconstruction(self, Psi, Phi, a):
        start = dt.now()
        r = np.zeros((self.patch_dim, self.batch_size, self.time_batch_size))
        #for t in range(self.time_batch_size):
            #size = min(self.timepoints - 1, t)
            #r[:,:,t] = np.einsum('pn,nct,cbt->pb', Phi, Psi[:,:,0:size+1], a[:,:,t::-1][:,:,0:size+1])
        #self.profile_print("get_reconstruction Calc", start)

        r2 = np.zeros((self.patch_dim, self.batch_size, self.time_batch_size))
        for t in range(self.time_batch_size):
            size = min(self.timepoints - 1, t)
            r2[:,:,t] = ten3dot2(Phi, Psi[:,:,0:size+1], a[:,:,t::-1][:,:,0:size+1])
        self.profile_print("get_reconstruction 2 Calc", start)

        #assert np.allclose(r, r2)

        return r2

    def a_cot(self, Psi, Phi, e):
        'Correlation over time'
        start = dt.now()
        result = np.zeros((self.cells, self.batch_size, self.time_batch_size))
        #for t in range(self.time_batch_size):
            #size = min(self.timepoints, self.time_batch_size - t)
            #result[:,:,t] = np.einsum('pbt,nct,pn->cb', e[:,:,t:t+size], Psi[:,:,t:t+size], Phi)
        #self.profile_print("dA Calc", start)

        start = dt.now()
        result2 = np.zeros((self.cells, self.batch_size, self.time_batch_size))
        for t in range(self.time_batch_size):
            size = min(self.timepoints, self.time_batch_size - t)
            result2[:,:,t] = ten3dot(e[:,:,t:t+size], Psi[:,:,0:size], Phi)
        self.profile_print("dA2 Calc", start)

        #assert np.allclose(result, result2)

        return result2

    def profile_print(self, msg, start):
        if not self.profile:
            return
        diff = dt.now() - start
        print '%20s | E=%s' % (msg, diff)

    def sparse_cost(self, a):
        if True:
            sigma = 1.00
            return (2 * (a/sigma)) / (1 + (a/sigma) ** 2)
        else:
            na = np.copy(a)
            na[a>0] = 1
            na[a<0] = -1
            return na

    def get_activity(self, a):
        max_active = self.batch_size * self.time_batch_size * self.neurons
        ac = np.copy(a)
        #cutoff = self.lambdav
        cutoff = 0.01
        ac[np.abs(ac) > cutoff] = 1
        ac[np.abs(ac) <= cutoff] = 0
        return 100 * np.sum(ac)/max_active

    def get_snr(self, VI, e):
        var = VI.var().mean()
        mse = (e ** 2).mean()
        return 10 * log(var/mse, 10)

    def draw(self, c, a, recon, VI):
        fg, ax = plt.subplots(3)
        ax[2].set_title('Activity')
        for i in range(self.neurons):
            ax[2].plot(range(self.time_batch_size), a[i,0,:])

        for t in range(self.time_batch_size):
            ax[0].imshow(np.reshape(recon[:,0,t], (self.sz, self.sz)), cmap = cm.binary, interpolation='nearest')
            ax[0].set_title('Recon iter=%d, t=%d' % (c, t))
            ax[1].imshow(np.reshape(VI[:,0,t], (self.sz, self.sz)), cmap = cm.binary, interpolation='nearest')
            ax[1].set_title('Image iter=%d, t=%d' % (c, t))

            plt.draw()
            #time.sleep(0.01)
        plt.close()

    def sparsify(self, VI, Psi, Phi):
        a = np.zeros((self.neurons, self.batch_size, self.time_batch_size))
        recon = self.get_reconstruction(Psi, Phi, a)
        e = VI - recon # error

        for c in range(self.citers):
            da = self.a_cot(Psi, Phi, e) - self.lambdav * self.sparse_cost(a)
            #da = self.a_cot(Psi, Phi, e)
            a += self.coeff_eta * da
            recon = self.get_reconstruction(Psi, Phi, a)
            e = VI - recon

            #if c > 30:
                #pdb.set_trace()
            if True or  c == self.citers or c % (self.citers/4) == 0:
                if self.visualizer:
                    self.draw(c, a, recon, VI)
                print '\t%d) SNR=%.2fdB, E=%.3f Activity=%.2f%%' % \
                    (c, self.get_snr(VI, e), np.sum(np.abs(e)), self.get_activity(a))

        return e, recon, a

    def train(self):
        if True:
            Psi = np.zeros((self.neurons, self.cells, self.timepoints))
            assert self.neurons == self.cells
            Psi[:,:,0] = np.eye(self.neurons)

            Phi = self.Phi

        for trial in range(self.num_trials):
            VI = self.load_videos()
            e, recon, a = self.sparsify(VI, Psi, Phi)

            print '%d) 1-SNR=%.2fdB' % (trial, self.get_snr(VI, e))

st = SpaceTime()
st.train()
