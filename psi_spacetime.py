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
from scipy import ndimage

dtype = theano.config.floatX
class PsiSpaceTime():

    # Parameters
    cells = 200
    timepoints = 7

    load_psi   = False
    save_psi   = True
    show_often = 5
    time_batch_size = 128
    batch_size = 10
    num_trials = 1000

    eta_init = 0.15
    eta_inc  = 32

    citers    = 20
    coeff_eta = 1.00
    #norm_Psi  = 0.43 # 3 * sqrt(var(data))
    lambdav   = 0.20

    data_name = 'Phi_169_45.0_IMAGES_DUCK'
    profile = True
    visualizer = False
    show = True

    # Inferred parameters
    graphics_initialized = False # Don't change

    def __init__(self):
        self.data = np.load('activity/activity_%s.npy' % self.data_name)
        self.neurons, self.patches, self.num_frames = np.shape(self.data)
        variances = np.zeros(self.num_frames - self.timepoints)
        for t in range(self.num_frames - self.timepoints):
            variances[t] = self.data[:,:,t:t+self.timepoints].var()
        self.norm_Psi = 3 * np.sqrt(variances.mean())
        self.norm_Psi /= 20
        #self.norm_Psi = 0.002
        plt.ion()

    def load_neurons(self):
        tbs = self.time_batch_size

        VN = np.zeros((self.neurons, self.batch_size, tbs))
        for b in range(self.batch_size):
            imi = np.floor((self.num_frames - tbs) * random.uniform(0, 1))
            pi = np.floor(self.patches * random.uniform(0,1))
            VN[:,b,:] = self.data[:,pi,imi:imi+tbs]
        return VN

    def normalize_Psi(self, Psi, a):
        start = dt.now()
        for i in range(self.cells):
            if a is not None:
                a_i = np.mean(a[i,:,:] ** 2)
                print '\t %d mean: %.4f' % (i, np.mean(a[i,:,:] ** 2))
                pdb.set_trace()
            else:
                a_i = self.norm_Psi
                #print '\t %d mean: %.4f' % (i,  np.mean(a[i,:,:] ** 2))
            Psi[:,i,:] *= self.norm_Psi/np.linalg.norm(Psi[:,i,:])
            #Psi[:,i,:] *= a_i/np.linalg.norm(Psi[:,i,:])
        self.profile_print('normPsi', start)
        return Psi

    def get_eta(self, trial):
        eta = self.eta_init
        for i in range(10):
            if trial < self.eta_inc * (i+1):
                return eta/self.batch_size
            eta /= 2.0
        return eta/self.batch_size

    def get_reconstruction(self, VN, Psi, a):
        r = np.zeros((self.neurons, self.batch_size, self.time_batch_size))
        start = dt.now()
        for t in range(self.time_batch_size):
            size = min(self.timepoints - 1, t)
            r[:,:,t] = tendot(Psi[:,:,0:size+1], a[:,:,t::-1][:,:,0:size+1])
        self.profile_print("get_reconstruction Calc", start)

        return r

    def a_cot(self, Psi, e):
        'Correlation over time'
        start = dt.now()
        result = np.zeros((self.cells, self.batch_size, self.time_batch_size))
        for t in range(self.time_batch_size):
            size = min(self.timepoints, self.time_batch_size - t)
            result[:,:,t] = ten2dot(Psi[:,:,0:size], e[:,:,t:t+size])
        self.profile_print("dA Calc", start)

        return result

    def psi_cot(self, a, e):
        'Correlation over time'
        start = dt.now()
        result = np.zeros((self.neurons, self.cells, self.timepoints))
        for tau in range(self.timepoints):
            for t in range(self.time_batch_size):
                if t+tau >= self.time_batch_size:
                    break
                result[:,:,tau] += np.tensordot(a[:,:,t], e[:,:,t+tau], [[1], [1]]).T
        self.profile_print("dPsi Calc", start)

        #assert np.allclose(result, result2)
        #print np.allclose(result, result2)

        return result / self.time_batch_size

    def profile_print(self, msg, start):
        if not self.profile:
            return
        diff = dt.now() - start
        print '%20s | E=%s' % (msg, diff)

    def sparse_cost(self, a):
        #sigma = 0.0001
        return 0
        if False:
            sigma = 1
            return (2 * (a/sigma)) / (1 + (a/sigma) ** 2)
        else:
            da = np.copy(a)
            da[da>0] = 1
            da[da<0] = -1
            return da * -1

    def get_activity(self, a):
        max_active = self.batch_size * self.time_batch_size * self.neurons
        ac = np.copy(a)
        #cutoff = self.lambdav
        cutoff = 0.01
        ac[np.abs(ac) > cutoff] = 1
        ac[np.abs(ac) <= cutoff] = 0
        return 100 * np.sum(ac)/max_active

    def get_snr(self, VN, e):
        var = VN.var()
        mse = (e ** 2).mean()
        return 10 * log(var/mse, 10)

    def draw(self, c, a, recon, VN):
        fg, ax = plt.subplots(3)
        ax[2].set_title('Activity')
        axis_height = np.max(np.abs(a))
        ax[2].axis([0, self.cells, -axis_height, axis_height])
        for i in range(self.cells):
            ax[2].plot(range(self.time_batch_size), a[i,0,:])

        #for t in range(self.time_batch_size):
            #ax[0].imshow(np.reshape(recon[:,0,t], (self.sz, self.sz)), cmap = cm.binary, interpolation='nearest')
            #ax[0].set_title('Recon iter=%d, t=%d' % (c, t))
            #ax[1].imshow(np.reshape(VN[:,0,t], (self.sz, self.sz)), cmap = cm.binary, interpolation='nearest')
            #ax[1].set_title('Image iter=%d, t=%d' % (c, t))

            plt.draw()
            #time.sleep(0.01)
        plt.show(block=True)
        plt.close()

    def sparsify(self, VN, Psi):
        a = np.zeros((self.cells, self.batch_size, self.time_batch_size))
        recon = self.get_reconstruction(VN, Psi, a)
        e = VN - recon # error

        for c in range(self.citers):
            da = self.a_cot(Psi, e) - self.lambdav * self.sparse_cost(a)
            a += self.coeff_eta * da
            recon = self.get_reconstruction(VN, Psi, a)
            e = VN - recon

            if c == self.citers - 1 or c % (self.citers/4) == 0:
                if self.visualizer and c == self.citers - 1:
                    self.draw(c, a, recon, VN)
                print '\t%d) SNR=%.2fdB, E=%.3f Activity=%.2f%%' % \
                    (c, self.get_snr(VN, e), np.sum(np.abs(e)), self.get_activity(a))

        return e, recon, a

    def train(self):
        if self.load_psi:
            Psi = np.load('psi_spacetime.npy')
        else:
            Psi = np.random.randn(self.neurons, self.cells, self.timepoints)
            Psi = self.normalize_Psi(Psi, None)

        for trial in range(self.num_trials):
            VN = self.load_neurons()
            e, recon, a = self.sparsify(VN, Psi)

            print '%d) 1-SNR=%.2fdB' % (trial, self.get_snr(VN, e))
            dPsi = self.psi_cot(a, e)
            Psi += self.get_eta(trial) * dPsi

            recon = self.get_reconstruction(VN, Psi, a)
            e = VN - recon
            Psi = self.normalize_Psi(Psi, a)

            if trial % self.show_often == 0:
                #self.showbfs(Psi)
                if self.save_psi:
                    np.save('psi_spacetime', Psi)
                    print 'Saved Psi'

    '''
    def reconstruct(self):
        Psi = np.load('psi_spacetime_phi.npy')

        self.batch_size = self.patch_per_dim ** 2 # Whole frame

        VN = self.load_video()
        a = np.zeros((self.neurons, self.batch_size, self.time_batch_size))
        for c in range(60):
            e = self.get_error(VN, Psi, a)
            print '\t%d) E=%.3f Activity=%.2f%%' % \
                (c, np.sum(np.abs(e)), self.get_activity(a))
            da = self.a_cot(Psi, e) - self.lambdav * self.sparse_cost(a)
            a += self.coeff_eta * da

        reconstruct = np.zeros((self.imsize, self.imsize, self.time_batch_size))
        for t in range(self.time_batch_size):
            I = VN[:,:,t]
            size = min(self.timepoints - 1, t)
            r_t = tendot(Psi[:,:,0:size+1], a[:,:,t::-1][:,:,0:size+1])

            i = 0
            for r in range(self.patch_per_dim):
                for c in range(self.patch_per_dim):
                    rr = r * self.sz
                    cc = c * self.sz
                    reconstruct[rr:rr+self.sz, cc:cc+self.sz, t] = \
                            np.reshape(r_t[:,i], (self.sz, self.sz))
                    i = i + 1

        np.save('reconstruct', reconstruct)

    def show_reconstruct(self):
        reconstruct = np.load('reconstruct.npy')
        for t in range(reconstruct.shape[2]):
            plt.imshow(reconstruct[:,:,t], cmap = cm.binary, interpolation='none')
            plt.title('Reconstruction t=%d' % t)
            plt.draw()
            time.sleep(0.5)

    def showbfs(self, Psi):
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

                maxA=max(abs(Psi[:,k,t])) # RESCALE
                img = reshape(Psi[:,k,t], (sz, sz), order='C').transpose()/maxA
                arr[index(i):index(i)+sz, index(j):index(j)+sz] = img

            plt.imshow(arr, cmap = cm.binary, interpolation='none')
            plt.title('Psi @ t=%d' % t)
            plt.draw()
            plt.savefig('Psi_%.2d.png' % t)
            plt.show()
        '''


pst = PsiSpaceTime()
pst.train()
