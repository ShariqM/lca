import matplotlib
matplotlib.rcParams.update({'figure.autolayout': True}) # Magical tight layout

import socket
if 'eweiss' in socket.gethostname():
    matplotlib.use('Agg') # Don't crash because $Display is not set correctly on the cluster

import traceback
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
    patch_dim = 64
    neurons   = 96
    timepoints = 5

    load_phi   = False
    save_phi   = False
    batch_size = 10
    time_batch_size = 64
    num_trials = 1000

    eta_init = 0.2
    eta_inc  = 10

    citers    = 40
    coeff_eta = 0.05
    lambdav   = 0.00

    data_name = 'IMAGES_DUCK_SHORT'
    profile = False
    visualizer = True
    show = True

    # Inferred parameters
    sz = int(np.sqrt(patch_dim))
    graphics_initialized = False # Don't change

    def __init__(self):
        self.images = scipy.io.loadmat('mat/%s.mat' % self.data_name)
        self.images = self.images[self.data_name]
        (self.imsize, imsize, self.num_images) = np.shape(self.images)
        self.patch_per_dim = int(np.floor(imsize / self.sz))
        plt.ion()


    def load_video(self):
        imsize, sz, tbs = self.imsize, self.sz, self.time_batch_size
        VI = np.zeros((self.patch_dim, self.batch_size, self.time_batch_size))
        for t in range(self.time_batch_size):
            i = 0
            for r in range(self.patch_per_dim):
                for c in range(self.patch_per_dim):
                    rr = r * sz
                    cc = c * sz
                    VI[:,i,t] = np.reshape(self.images[rr:rr+sz, cc:cc+sz, t], self.patch_dim, 1)
                    i = i + 1
        return VI

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
        #print 'norm 0', np.linalg.norm(Phi[:,0,:])
        Phi = np.einsum('pnt,nn->pnt', Phi, np.diag(np.average(np.sum(a ** 2, axis=2), axis=1)))
        #print 'norm 0 after', np.linalg.norm(Phi[:,0,:])
        #for i in range(self.timepoints):
            #Phi[:,:,i] = np.dot(Phi[:,:,i], np.diag(1/np.sqrt(np.sum(Phi[:,:,i]**2, axis = 0))))
        return Phi

    def get_eta(self, trial):
        eta = self.eta_init
        for i in range(10):
            if trial < self.eta_inc * (i+1):
                return eta/self.batch_size
            eta /= 2.0
        return eta/self.batch_size

    def get_error(self, VI, Phi, a):
        e = np.zeros((self.patch_dim, self.batch_size, self.time_batch_size))
        start = dt.now()
        for t in range(self.time_batch_size):
            I = VI[:,:,t]
            size = min(self.timepoints - 1, t)
            e[:,:,t] = I - tendot(Phi[:,:,0:size+1], a[:,:,t::-1][:,:,0:size+1])

        self.profile_print("get_error2 Calc", start)

        return e

    def a_cot(self, Phi, e):
        'Correlation over time'
        start = dt.now()
        result = np.zeros((self.neurons, self.batch_size, self.time_batch_size))
        for t in range(self.time_batch_size):
            size = min(self.timepoints, self.time_batch_size - t)
            result[:,:,t] = ten2dot(Phi[:,:,0:size], e[:,:,t:t+size])
        self.profile_print("dA2 Calc", start)

        #start = dt.now()
        #result = np.zeros((self.neurons, self.batch_size, self.time_batch_size))
        #for t in range(self.time_batch_size):
            #size = min(self.timepoints, self.time_batch_size - t)
            #result[:,:,t] = np.einsum('pnt,pbt->nb', Phi[:,:,0:size], e[:,:,t:t+size])
        #self.profile_print("dA Calc", start)
        return result

    def phi_cot(self, a, e):
        'Correlation over time'
        start = dt.now()
        result = np.zeros((self.patch_dim, self.neurons, self.timepoints))
        for tau in range(self.timepoints):
            for t in range(self.time_batch_size):
                if t+tau >= self.time_batch_size:
                    break
                result[:,:,tau] += np.einsum('nb,pb->pn', a[:,:,t], e[:,:,t+tau])
        self.profile_print("dPhi Calc", start)

        return result / self.time_batch_size

    def profile_print(self, msg, start):
        if not self.profile:
            return
        diff = dt.now() - start
        print '%20s | E=%s' % (msg, diff)

    def sparse_cost(self, a):
        return (2 * a) / (1 + a ** 2)

    def get_activity(self, a):
        max_active = self.batch_size * self.time_batch_size * self.neurons
        ac = np.copy(a)
        #cutoff = self.lambdav
        cutoff = 0.01
        ac[np.abs(ac) > cutoff] = 1
        ac[np.abs(ac) <= cutoff] = 0
        return 100 * np.sum(ac)/max_active

    def get_activity(self, a):
        max_active = self.batch_size * self.time_batch_size * self.neurons
        return np.sum(np.abs(a))/max_active

    def get_snr(self, VI, e):
        snr_sum = 0.0
        for t in range(self.time_batch_size):
            I = VI[:,:,t]
            R = I - e[:,:,t]
            var = I.var().mean()
            mse = (R ** 2).mean()
            snr_sum += 10 * math.log(var/mse, 10)
        return snr_sum/self.time_batch_size

    def sparsify(self, VI, Phi):
        a = np.zeros((self.neurons, self.batch_size, self.time_batch_size))
        e = self.get_error(VI, Phi, a)
        recon = VI - e

        if self.visualizer:
            if not self.graphics_initialized:
                self.graphics_initialized = True
                fg, self.ax = plt.subplots(2)

            for t in range(self.time_batch_size):
                self.ax[0].imshow(np.reshape(recon[:,0,t], (self.sz, self.sz)), cmap = cm.binary, interpolation='nearest')
                self.ax.set_title('Recon t=%d' % t)
                self.ax[1].imshow(np.reshape(VI[:,0,t], (self.sz, self.sz)), cmap = cm.binary, interpolation='nearest')
                self.ax.set_title('Image t=%d' % t)
                plt.draw()
                time.sleep(0.05)

        for c in range(self.citers):
            if c == self.citers or c % (self.citers/4) == 0:
                print '\t%d) 1-SNR=%.2fdB' % (c, self.get_snr(VI, e))
                print '\t%d) E=%.3f Activity=%.2f%%' % \
                    (c, np.sum(np.abs(e)), self.get_activity(a))

            da = self.a_cot(Phi, e) - self.lambdav * self.sparse_cost(a)
            a += self.coeff_eta * da
            e = self.get_error(VI, Phi, a)
            recon = VI - e

            if self.visualizer:
                for t in range(self.time_batch_size):
                    self.ax[0].imshow(np.reshape(recon[:,0,t], (self.sz, self.sz)), cmap = cm.binary, interpolation='nearest')
                    self.ax.set_title('Recon t=%d' % t)
                    self.ax[1].imshow(np.reshape(VI[:,0,t], (self.sz, self.sz)), cmap = cm.binary, interpolation='nearest')
                    self.ax.set_title('Image t=%d' % t)
                    plt.draw()
                    time.sleep(0.05)

        return e, a

    def train(self):
        if self.load_phi:
            Phi = np.load('spacetime_phi.npy')
        else:
            Phi = np.random.randn(self.patch_dim, self.neurons, self.timepoints)
            if False:
                Phi[:,:,0] = np.dot(Phi[:,:,0], np.diag(1/np.sqrt(np.sum(Phi[:,:,0]**2, axis = 0))))
                for i in range(1, self.timepoints):
                    Phi[:,:,i] = Phi[:,:,i-1] + 0.1 * np.random.randn(self.patch_dim, self.neurons)
                    Phi[:,:,i] = np.dot(Phi[:,:,i], np.diag(1/np.sqrt(np.sum(Phi[:,:,i]**2, axis = 0))))
            else:
                for i in range(self.timepoints):
                    Phi[:,:,i] = np.dot(Phi[:,:,i], np.diag(1/np.sqrt(np.sum(Phi[:,:,i]**2, axis = 0))))

        for trial in range(self.num_trials):
            VI = self.load_videos()
            e, a = self.sparsify(VI, Phi)

            print '%d) E1=%.3f' % (trial, np.sum(np.abs(e)))
            print '%d) 1-SNR=%.2fdB' % (trial, self.get_snr(VI, e))
            dPhi = self.phi_cot(a, e)
            Phi += self.get_eta(trial) * dPhi

            e = self.get_error(VI, Phi, a)
            print '%d) 2-SNR=%.2fdB' % (trial, self.get_snr(VI, e))
            print '%d) E2=%.3f' % (trial, np.sum(np.abs(e)))
            Phi = self.normalize_Phi(Phi, a)

            self.showbfs(Phi)
            if self.save_phi:
                np.save('spacetime_phi', Phi)

    def reconstruct(self):
        Phi = np.load('spacetime_phi.npy')

        self.batch_size = self.patch_per_dim ** 2 # Whole frame

        VI = self.load_video()
        a = np.zeros((self.neurons, self.batch_size, self.time_batch_size))
        for c in range(60):
            e = self.get_error(VI, Phi, a)
            print '\t%d) E=%.3f Activity=%.2f%%' % \
                (c, np.sum(np.abs(e)), self.get_activity(a))
            da = self.a_cot(Phi, e) - self.lambdav * self.sparse_cost(a)
            a += self.coeff_eta * da

        reconstruct = np.zeros((self.imsize, self.imsize, self.time_batch_size))
        for t in range(self.time_batch_size):
            I = VI[:,:,t]
            size = min(self.timepoints - 1, t)
            r_t = tendot(Phi[:,:,0:size+1], a[:,:,t::-1][:,:,0:size+1])

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
            plt.savefig('Phi_%d.png' % t)
            plt.show()


st = SpaceTime()
st.train()
#st.reconstruct()
#st.show_reconstruct()
