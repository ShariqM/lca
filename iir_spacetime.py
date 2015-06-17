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
from numpy import tensordot as tdot

dtype = theano.config.floatX
class IIRSpaceTime():

    # Parameters
    patch_dim = 16
    neurons   = patch_dim * 1
    cells     = patch_dim
    #patch_dim = 64
    #neurons   = patch_dim * 4
    #cells     = patch_dim

    load_phi   = False
    save_phi   = False
    batch_size = 2
    time_batch_size = 8
    #batch_size = 10
    #time_batch_size = 64
    num_trials = 1000

    eta_init = 0.6
    eta_inc  = 10

    citers    = 10
    coeff_eta = 5e-3
    lambdav   = 1.00

    data_name = 'IMAGES_DUCK_SHORT'
    profile = True
    visualizer = False
    show = True

    # Inferred parameters
    sz = int(np.sqrt(patch_dim))
    graphics_initialized = False # Always leave as False

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
        norm0 = np.linalg.norm(Phi[:,0,:])
        for i in range(self.neurons):
            #Phi[:,i,:] *= 0.02/np.linalg.norm(Phi[:,i,:])
            Phi[:,i,:] *= 0.1/np.linalg.norm(Phi[:,i,:])
            #Phi[:,i,:] *= np.mean(a[i,:,:] ** 2)
            #print np.mean(a[i,:,:] ** 2)
        print 'Norm Transfer %.8f->%.8f' % (norm0, np.linalg.norm(Phi[:,0,:]))
        return Phi

    def get_eta(self, trial):
        eta = self.eta_init
        for i in range(10):
            if trial < self.eta_inc * (i+1):
                return eta/self.batch_size
            eta /= 2.0
        return eta/self.batch_size

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

    def get_snr(self, VI, e):
        #recon = np.random.randn(self.patch_dim, self.batch_size, self.time_batch_size)
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

    def grad_M(self, Phi, error, u):
        start = dt.now()
        result = np.tensordot(np.tensordot(error, Phi, [[0], [0]]), u, [[0,1], [1,2]])
        self.profile_print("dM2 Calc", start)
        return result

    def grad_a(self, error, Phi, M, B):
        start = dt.now()
        result = np.zeros((self.cells, self.batch_size, self.time_batch_size))

        I = np.eye(self.neurons)
        iplusm = np.zeros((self.neurons, self.neurons, self.time_batch_size))
        iplusm[:,:,0] = I
        for t in range(1, self.time_batch_size):
            iplusm[:,:,t] = np.dot(iplusm[:,:,t-1], I+M)

        start = dt.now()
        for TT in range(self.time_batch_size - 1):
            tmp = iplusm[:,:,:self.time_batch_size-TT-1]
            x = tdot(error[:,:,TT+1:], Phi, [[0], [0]])
            y = tdot(tmp, B, [[1], [0]])
            result[:,:,TT] = tdot(x, y, [[1,2], [1,0]]).T

        self.profile_print("dA Calc2", start)
        return result

    def get_reconstruction(self, VI, Phi, M, B, a):
        start = dt.now()
        u = np.zeros((self.neurons, self.batch_size, self.time_batch_size))
        I = np.eye(self.neurons)
        u[:,:,0] = np.dot(B, a[:,:,0])
        for t in range(1, self.time_batch_size):
            u[:,:,t] = np.dot((I+M), u[:,:,t-1]) + np.dot(B, a[:,:,t])
        r = np.tensordot(Phi, u, [[1], [0]])
        self.profile_print("get_reconstruction Calc", start)
        return u, r

    def sparsify(self, VI, Phi, M, B):
        #a = np.zeros((self.neurons, self.batch_size, self.time_batch_size))
        #a = np.zeros((self.neurons, self.batch_size, self.time_batch_size))
        a = 1.0 * np.random.randn(self.neurons, self.batch_size, self.time_batch_size)
        u, recon = self.get_reconstruction(VI, Phi, M, B, a)
        error = VI - recon

        print '\t%d) SNR=%.2fdB, E=%.3f Activity=%.2f%%' % \
            (-1, self.get_snr(VI, error), np.sum(np.abs(error)), self.get_activity(a))

        for c in range(self.citers):
            #da = self.a_cot(Phi, e) - self.lambdav * self.sparse_cost(a)
            da = self.grad_a(error, Phi, M, B)
            a += self.coeff_eta * da

            u, recon = self.get_reconstruction(VI, Phi, M, B, a)
            error = VI - recon
            #pdb.set_trace()

            #if c == self.citers or c % (self.citers/4) == 0:
            if True:
                if self.visualizer:
                    self.draw(c, a, recon, VI)
                print '\t%d) SNR=%.2fdB, E=%.3f Activity=%.2f%%' % \
                    (c, self.get_snr(VI, error), np.sum(np.abs(error)), self.get_activity(a))

        self.coeff_eta = 0.25
        return error, recon, a

    def train(self):
        Phi = np.random.randn(self.patch_dim, self.neurons)
        Phi = np.dot(Phi, np.diag(1/np.sqrt(np.sum(Phi**2, axis = 0))))

        B = 0.1 * np.random.randn(self.neurons, self.cells)
        M = 0.1 * np.random.randn(self.neurons, self.neurons)

        for trial in range(self.num_trials):
            VI = self.load_videos()
            error, recon, a = self.sparsify(VI, Phi, M, B)
            pdb.set_trace()

            print '%d) 1-SNR=%.2fdB' % (trial, self.get_snr(VI, error))
            u, recon = self.get_reconstruction(VI, Phi, M, B, a)

            grad_M = self.grad_M(Phi, error, u)
            M += 0.001 * self.get_eta(trial) * grad_M
            u, recon = self.get_reconstruction(VI, Phi, M, B, a)
            error = VI - recon

            print '%d) 2-SNR=%.2fdB' % (trial, self.get_snr(VI, error))
            pdb.set_trace()
            #dPhi = self.grad_phi(error, Phi, M, B)
            #Phi += self.get_eta(trial) * dPhi

            #u, recon = self.get_reconstruction(VI, Phi, M, B, a)
            #e = VI - recon
            #print '%d) 3-SNR=%.2fdB' % (trial, self.get_snr(VI, e))
            #Phi = self.normalize_Phi(Phi, a)

            #self.showbfs(Phi)
            if self.save_phi:
                np.save('iir_spacetime_phi', Phi)

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

st = IIRSpaceTime()
st.train()
