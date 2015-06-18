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
from iir_helpers import *
import sys
import random
from numpy import tensordot as tdot

dtype = theano.config.floatX
class IIRSpaceTime():

    # Parameters
    #patch_dim = 16
    #neurons   = patch_dim * 2
    #cells     = patch_dim
    patch_dim = 64
    neurons   = patch_dim * 4
    cells     = patch_dim

    load_phi   = False
    save_phi   = False
    batch_size = 20
    time_batch_size = 64
    #batch_size = 10
    #time_batch_size = 64
    num_trials = 1000

    Phi_eta_init = 0.02
    M_eta_init   = 0.002 # Depend on the dim on input
    B_eta_init   = 0.002 # Depend on the dim on input

    eta_inc  = 200

    Phi_norm = 1
    M_norm   = 0.01
    B_norm   = 0.1

    citers    = 40
    coeff_eta = 5e-2
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

    def get_eta(self, eta_init, trial):
        eta = eta_init
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
        x = tdot(error[:,:,1:], Phi, [[0], [0]])
        result = tdot(x, u[:,:,:-1], [[0,1], [1,2]])
        self.profile_print("dM Calc", start)
        return result

    def grad_Phi(self, error, u):
        start = dt.now()
        result = tdot(error, u, [[1,2], [1,2]])
        self.profile_print("dPhi Calc", start)
        return result

    def grad_B(self, error, Phi, a):
        start = dt.now()
        x = tdot(error[:,:,1:], Phi, [[0], [0]])
        result = tdot(x, a[:,:,:-1], [[0,1], [1,2]])
        return result

    def debug_a(self, error, Phi, M, B, iplusm):
        result = np.zeros((self.cells, self.batch_size, self.time_batch_size))
        start = dt.now()
        for TT in range(self.time_batch_size):
            for JJ in range(self.cells):
                for BB in range(self.batch_size):
                    for t in range(TT+1, self.time_batch_size):
                        tmp_t = 0
                        for k in range(self.patch_dim):
                            tmp_k = 0
                            for i in range(self.neurons):
                                tmp_i = 0
                                for l in range(self.neurons):
                                    tmp_i += iplusm[i, l, t-1-TT] * B[l, JJ]
                                tmp_k += Phi[k, i] * tmp_i
                            tmp_t += error[k, BB, t] * tmp_k
                        result[JJ, BB, TT] += tmp_t
        return result

    def grad_a(self, error, Phi, M, B, debug=False):
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

        self.profile_print("dA Calc", start)

        start = dt.now()
        for TT in range(self.time_batch_size - 1):
            tmp = iplusm[:,:,:self.time_batch_size-TT-1]
            result[:,:,TT] = grad_a_TT(error[:,:,TT+1:], Phi, tmp, B)
        self.profile_print("dA Calc2", start)

        if debug:
            r2 = self.debug_a(error, Phi, M, B, iplusm)
            print np.max(result - r2)
            assert np.allclose(result, r2)

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

    def sparsify(self, VI, Phi, M, B, debug=False):
        #a = np.zeros((self.cells, self.batch_size, self.time_batch_size))
        #a = np.zeros((self.cells, self.batch_size, self.time_batch_size))
        a = 1.0 * np.random.randn(self.cells, self.batch_size, self.time_batch_size)
        u, recon = self.get_reconstruction(VI, Phi, M, B, a)
        error = VI - recon

        print '\t%d) SNR=%.2fdB, E=%.3f Activity=%.2f%%' % \
            (-1, self.get_snr(VI, error), np.sum(np.abs(error)), self.get_activity(a))

        for c in range(self.citers):
            #da = self.a_cot(Phi, e) - self.lambdav * self.sparse_cost(a)
            da = self.grad_a(error, Phi, M, B, debug)
            a += self.coeff_eta * da

            u, recon = self.get_reconstruction(VI, Phi, M, B, a)
            error = VI - recon

            if c == self.citers or c % (self.citers/4) == 0:
            #if True:
                if self.visualizer:
                    self.draw(c, a, recon, VI)
                print '\t%d) SNR=%.2fdB, E=%.3f Activity=%.2f%%' % \
                    (c, self.get_snr(VI, error), np.sum(np.abs(error)), self.get_activity(a))

        return error, recon, a

    def norm_Phi(self, Phi):
        return np.dot(Phi, np.diag(self.Phi_norm/np.sqrt(np.sum(Phi**2, axis = 0))))

    def norm_B(self, B):
        return np.dot(B, np.diag(self.B_norm/np.sqrt(np.sum(B**2, axis = 0))))

    def norm_M(self, M):
        return np.dot(M, np.diag(self.M_norm/np.sqrt(np.sum(M**2, axis = 0))))

    def train(self):
        Phi = np.random.randn(self.patch_dim, self.neurons)
        Phi = self.norm_Phi(Phi)

        B = 0.1 * np.random.randn(self.neurons, self.cells)
        B = self.norm_B(B)
        M = 0.1 * np.random.randn(self.neurons, self.neurons)
        M = self.norm_M(M)

        for trial in range(self.num_trials):
            VI = self.load_videos()
            #error, recon, a = self.sparsify(VI, Phi, M, B, debug=(trial > 0))
            error, recon, a = self.sparsify(VI, Phi, M, B)

            print '%d) 1-SPARSIFY-SNR=%.2fdB' % (trial, self.get_snr(VI, error))
            u, recon = self.get_reconstruction(VI, Phi, M, B, a)

            grad_M = self.grad_M(Phi, error, u)
            M += self.get_eta(self.M_eta_init, trial) * grad_M
            u, recon = self.get_reconstruction(VI, Phi, M, B, a)
            error = VI - recon

            print '%d) 2-GRAD_M-SNR=%.2fdB' % (trial, self.get_snr(VI, error))
            M = self.norm_M(M)
            u, recon = self.get_reconstruction(VI, Phi, M, B, a)
            print '%d) 3-NORM_M-SNR=%.2fdB' % (trial, self.get_snr(VI, error))

            grad_B = self.grad_B(error, Phi, a)
            B += self.get_eta(self.B_eta_init, trial) * grad_B
            u, recon = self.get_reconstruction(VI, Phi, M, B, a)
            error = VI - recon
            print '%d) 4-GRAD_B-SNR=%.2fdB' % (trial, self.get_snr(VI, error))
            B = self.norm_B(B)
            u, recon = self.get_reconstruction(VI, Phi, M, B, a)
            print '%d) 5-NORM_B-SNR=%.2fdB' % (trial, self.get_snr(VI, error))

            grad_Phi = self.grad_Phi(error, u)
            Phi += self.get_eta(self.Phi_eta_init, trial) * grad_Phi
            u, recon = self.get_reconstruction(VI, Phi, M, B, a)
            error = VI - recon
            print '%d) 5-GRAD_Phi-SNR=%.2fdB' % (trial, self.get_snr(VI, error))
            Phi = self.norm_Phi(Phi)
            u, recon = self.get_reconstruction(VI, Phi, M, B, a)
            print '%d) 6-NORM_Phi-SNR=%.2fdB' % (trial, self.get_snr(VI, error))


            #dPhi = self.grad_phi(error, Phi, M, B)
            #Phi += self.get_eta(trial) * dPhi

            #u, recon = self.get_reconstruction(VI, Phi, M, B, a)
            #e = VI - recon
            #print '%d) 3-SNR=%.2fdB' % (trial, self.get_snr(VI, e))
            #Phi = self.normalize_Phi(Phi, a)
            #if trial > 3:
                #pdb.set_trace()

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
