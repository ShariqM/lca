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
    cells      = 400
    timepoints = 7

    load_psi   = False
    save_psi   = True
    psi_identity = False
    save_often = 5
    batch_size = 10
    time_batch_size = 128
    num_trials = 1000

    eta_init = 0.10
    eta_inc  = 32

    sigma = 0.1
    citers    = 561
    coeff_eta = 0.01
    #coeff_eta = 1e-3
    lambdav   = 0.20
    alpha     = 0.1
    sparse_cutoff = 0.05

    data_name = 'IMAGES_DUCK_SHORT'
    phi_name = 'Phi_169_45.0'
    #phi_name = 'Phi_463/Phi_463_0.3'
    profile = True
    visualizer = False
    show = True

    # Other
    sz = int(np.sqrt(patch_dim))
    graphics_initialized = False # Don't change
    psi_name = 'dict/cul/169/cul_spacetime_169_7'

    def __init__(self):
        self.images = scipy.io.loadmat('mat/%s.mat' % self.data_name)
        self.images = self.images[self.data_name]
        (self.imsize, imsize, self.num_images) = np.shape(self.images)
        self.patch_per_dim = int(np.floor(imsize / self.sz))

        if self.psi_identity:
            self.coeff_eta = 0.1
            self.citers = 61

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

    def get_eta(self, trial):
        eta = self.eta_init
        for i in range(10):
            if trial < self.eta_inc * (i+1):
                return eta/self.batch_size
            eta /= 2.0
        return eta/self.batch_size

    def find_nearest(a, a0):
        "Element in nd array `a` closest to the scalar value `a0`"
        idx = np.abs(a - a0).argmin()
        return a.flat[idx]

    def get_reconstruction(self, Psi, Phi, a, Ups=None):
        start = dt.now()
        if Ups is None:
            Ups = np.tensordot(Phi, Psi, [[1], [0]])
            Ups = np.reshape(Ups, (self.patch_dim, self.cells * self.timepoints))

        for i in range(3):
            ac = np.copy(a)
            ac = np.swapaxes(ac, 0, 1)

            if i == 0:
                start = dt.now()
                ahat = np.zeros((self.batch_size, self.cells * self.timepoints,
                                 self.time_batch_size))
                ac = ac[:,:,-1::-1] # Reverse
                ac = np.reshape(ac, (self.batch_size, self.cells * self.time_batch_size))
                #dac = np.zeros((self.batch_size, self.cells * self.time_batch_size))
                #x = 0
                #for r in range(self.cells):
                    #for q in range(self.time_batch_size):
                        #dac[:,x] = ac[:,r,q]
                        #x += 1
                #ac = np.copy(dac)
                x = 0
                for t in range(self.time_batch_size):
                    x = x + 1
                print x
                self.profile_print("get_reconstruction iter1 Calc", start)

                for t in range(self.timepoints):
                    act = np.zeros((self.batch_size, self.cells * self.timepoints))
                    idx = self.time_batch_size-t-1
                    act[:,0:self.cells*(t+1)] = ac[:,self.cells * idx : self.cells * (idx+t+1)]
                    ahat[:,:,t] = act

                for t in range(self.timepoints, self.time_batch_size):
                    act = np.zeros((self.batch_size, self.cells * self.timepoints))
                    size = self.timepoints - 1

                    idx = self.time_batch_size-t-1
                    act[:,0:self.cells*(size+1)] = ac[:,self.cells * idx : self.cells * (idx+size+1)]
                    ahat[:,:,t] = act
                    #if t == 0:
                        #pdb.set_trace()
                self.profile_print("get_reconstruction loop3 1 Calc", start)
            elif i == 1:
                start = dt.now()
                ahat2 = np.zeros((self.batch_size, self.cells * self.timepoints,
                                 self.time_batch_size))
                ac = ac[:,:,-1::-1] # Reverse
                for t in range(self.time_batch_size):
                    act = np.zeros((self.batch_size, self.cells, self.timepoints))
                    size = min(self.timepoints - 1, t)
                    idx = self.time_batch_size-t-1
                    act[:,:,0:size+1] = ac[:,:,idx:idx+size+1]
                    ahat2[:,:,t] = np.reshape(act, (self.batch_size,
                                                   self.cells * self.timepoints))
                    #if t == 0:
                        #pdb.set_trace()
                self.profile_print("get_reconstruction loop3 2 Calc", start)
            else:
                start = dt.now()
                ahat3 = np.zeros((self.batch_size, self.cells * self.timepoints,
                                 self.time_batch_size))

                for t in range(self.time_batch_size):
                    act = np.zeros((self.batch_size, self.cells, self.timepoints))
                    size = min(self.timepoints - 1, t)
                    act[:,:,0:size+1] = ac[:,:,t::-1][:,:,0:size+1]
                    ahat3[:,:,t] = np.reshape(act, (self.batch_size,
                                                   self.cells * self.timepoints))
                self.profile_print("get_reconstruction loop3 3 Calc", start)

                print '1,2', np.allclose(ahat, ahat2, atol=1e-4)
                print '2,3', np.allclose(ahat2, ahat3, atol=1e-4)
                pdb.set_trace()

        self.profile_print("get_reconstruction loop Calc", start)
        #r = np.tensordot(Ups, ahat, [[1], [1]])
        r = tconv(Ups, ahat)
        self.profile_print("get_reconstruction Calc", start)

        return r

    def a_cot(self, Ups, e):
        'Correlation over time'
        start = dt.now()
        ec = np.copy(e)
        ec = np.swapaxes(ec, 0,1)
        error = np.zeros((self.batch_size, self.patch_dim * self.timepoints, self.time_batch_size))
        start = dt.now()
        for t in range(self.time_batch_size):
            ect = np.zeros((self.batch_size, self.patch_dim, self.timepoints))
            size = min(self.timepoints, self.time_batch_size - t)
            ect[:,:,0:size] = ec[:,:,t:t+size]
            error[:,:,t] = np.reshape(ect, (self.batch_size, self.patch_dim * self.timepoints))
        self.profile_print("dA loop", start)

        #result = np.tensordot(Ups, error, [[1], [1]])
        result = tconv(Ups, error)
        self.profile_print("dA Calc", start)

        return result

    def profile_print(self, msg, start):
        if not self.profile:
            return
        diff = dt.now() - start
        print '%20s | E=%s' % (msg, diff)

    def sparse_cost(self, a):
        if True:
            return (2 * (a/self.sigma)) / (1 + (a/self.sigma) ** 2)
        else:
            na = np.copy(a)
            na[a>0] = 1
            na[a<0] = -1
            return na

    def get_avg(self, a):
        return np.mean(np.abs(a))

    def get_activity(self, a):
        max_active = self.batch_size * self.time_batch_size * self.cells
        ac = np.copy(a)
        ac[np.abs(ac) > self.sparse_cutoff] = 1
        ac[np.abs(ac) <= self.sparse_cutoff] = 0
        return 100 * np.sum(ac)/max_active

    def get_snr(self, VI, e):
        var = VI.var().mean()
        mse = (e ** 2).mean()
        return 10 * log(var/mse, 10)

    def get_snr_t(self, VI, Psi, Phi, a):
        ta = np.copy(a)
        ta[np.abs(ta) < self.sparse_cutoff] = 0
        recon = self.get_reconstruction(Psi, Phi, ta)
        e = VI - recon
        return self.get_snr(VI, e)

    def draw(self, c, a, recon, VI):
        fg, ax = plt.subplots(3)
        ax[2].set_title('Activity')
        for i in range(self.cells):
            ax[2].plot(range(self.time_batch_size), a[i,0,:])

        for t in range(self.time_batch_size):
            ax[0].imshow(np.reshape(recon[:,0,t], (self.sz, self.sz)), cmap = cm.binary, interpolation='nearest')
            ax[0].set_title('Recon iter=%d, t=%d' % (c, t))
            ax[1].imshow(np.reshape(VI[:,0,t], (self.sz, self.sz)), cmap = cm.binary, interpolation='nearest')
            ax[1].set_title('Image iter=%d, t=%d' % (c, t))

            plt.draw()
            #time.sleep(0.01)
        plt.close()

    def thresh(self, a):
        a[np.abs(a) < self.sparse_cutoff] = 0
        return a

    def compute_Ups(self, Psi, Phi, ctype=1):
        Ups = np.tensordot(Phi, Psi, [[1], [0]]) # Precompute Upsilon
        if ctype == 1:
            Ups = np.swapaxes(Ups, 0, 1)
            Ups = np.reshape(Ups, (self.cells, self.patch_dim * self.timepoints))
        else:
            Ups = np.reshape(Ups, (self.patch_dim, self.cells * self.timepoints))
        return Ups

    def sparsify(self, VI, Psi, Phi):
        a = np.zeros((self.cells, self.batch_size, self.time_batch_size))
        recon = self.get_reconstruction(Psi, Phi, a)
        e = VI - recon # error
        Ups = self.compute_Ups(Psi, Phi)
        Ups_2 = self.compute_Ups(Psi, Phi, ctype=2)

        for c in range(self.citers):
            start = dt.now()
            da = self.a_cot(Ups, e) - self.lambdav * self.sparse_cost(a)
            #da = self.a_cot(Psi, Phi, e)
            a += self.coeff_eta * da
            recon = self.get_reconstruction(Psi, Phi, a, Ups_2)
            e = VI - recon

            if c == self.citers - 1 or c % (self.citers/10) == 0:
                if self.visualizer:
                    self.draw(c, a, recon, VI)
                print '\t%d) SNR=%.2fdB, SNR_T=%.2fdB, E=%.3f Activity=%.2f%% Avg=%.2f' % \
                    (c, self.get_snr(VI, e), self.get_snr_t(VI, Psi, Phi, a), \
                     np.sum(np.abs(e)), self.get_activity(a), self.get_avg(a))
            self.profile_print("Sparse iter", start)

        return e, recon, a

    def normalize_Psi(self, Psi, a):
        start = dt.now()
        for i in range(self.cells):
            if a is not None:
                a_i = (np.mean(a[i,:,:] ** 2)) / (self.sigma ** 2)
                #print '\t %d mean: %.4f' % (i, np.mean(a[i,:,:] ** 2))
                a_i = a_i ** self.alpha
            else:
                a_i = self.norm_Psi
            Psi[:,i,:] *= a_i/np.linalg.norm(Psi[:,i,:])
        self.profile_print('normPsi', start)
        return Psi

    def psi_cot(self, a, Phi, e):
        'Correlation over time'
        start = dt.now()
        result = np.zeros((self.neurons, self.cells, self.timepoints))

        #a = np.copy(a)
        #a = self.thresh(a)
        for tau in range(self.timepoints):
            for t in range(self.time_batch_size):
                if t+tau >= self.time_batch_size:
                    break
                result[:,:,tau] += ten_2_2_2(a[:,:,t], Phi, e[:,:,t+tau])

        self.profile_print("dPsi Calc", start)

        return result / self.time_batch_size

    def train(self):
        if self.load_psi:
            Psi = np.load(self.psi_name)
            if Psi.shape != (self.nuerons, self.cells, self.timepoints):
                raise Exception("Incompatible Psi loaded")
            Phi = self.Phi
        else:
            if not self.psi_identity:
                Psi = 0.1 * np.random.randn(self.neurons, self.cells, self.timepoints)
                for t in range(self.timepoints):
                    Psi[:,:,t] = np.dot(Psi[:,:,t], np.diag(1/np.sqrt(np.sum(Psi[:,:,t]**2, axis=0))))
            #elif not self.psi_identity:
                #Psi = np.zeros((self.neurons, self.cells, self.timepoints))
                #for j in range(self.cells):
                    #i = random.randint(0, self.neurons - 1)
                    #act = [0, 0.2, 0.4, 0.6, 0.4, 0.2, 0]
                    #for t in range(self.timepoints):
                        #Psi[i,j,t] = act[t]
            else:
                Psi = np.zeros((self.neurons, self.cells, self.timepoints))
                assert self.neurons == self.cells
                Psi[:,:,self.timepoints/2] = np.eye(self.neurons)

            Phi = self.Phi

        for trial in range(self.num_trials):
            VI = self.load_videos()
            e, recon, a = self.sparsify(VI, Psi, Phi)

            print '%d) 1-SNR=%.2fdB' % (trial, self.get_snr(VI, e))
            dPsi = self.psi_cot(a, Phi, e)
            Psi += self.get_eta(trial) * dPsi

            recon = self.get_reconstruction(Psi, Phi, a)
            e = VI - recon
            print '%d) 2-SNR=%.2fdB' % (trial, self.get_snr(VI, e))
            Psi = self.normalize_Psi(Psi, a)

            if trial % self.save_often == 0:
                if self.save_psi:
                    np.save(self.psi_name + ("_%.3f.npy" % (float(trial)/self.num_trials)), Psi)
                    print 'Saved Psi'

st = SpaceTime()
st.train()
