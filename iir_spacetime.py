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

    LOG_NAME = 'iir_log.txt'

    # Parameters
    #patch_dim = 16
    #neurons   = patch_dim * 2
    #cells     = patch_dim
    patch_dim = 144
    neurons   = 4
    cells     = 4

    load_phi   = None
    save_phi   = None
    log_and_save = False

    batch_size = 20
    time_batch_size = 15
    #batch_size = 10
    #time_batch_size = 64
    thresh_type  = 'soft'
    num_trials = 10000

    coeff_test = True # Learn easy dynamics (IMAGES_PHI_INTERP)
    M_backprop_steps = 1

    Phi_eta_init = 1.2  / time_batch_size
    M_eta_init   = (0.02 / (M_backprop_steps)) / time_batch_size # Depend on the dim on input?
    B_eta_init   = 0.02 / time_batch_size # Depend on the dim on input?

    eta_inc  = 500

    Phi_norm = 1
    M_norm   = 0.01 # Not running except for on init
    B_norm   = 0.1

    citers    = 446
    coeff_eta = 5e-2
    #coeff_eta = 1e-1
    lambdav   = 0.394

    data_name = 'IMAGES_PHI_463_INTERP'
    profile = False
    visualizer = True
    show = True

    # Inferred parameters
    sz = int(np.sqrt(patch_dim))
    graphics_initialized = False # Always leave as False

    def log(self, Phi, B, M, write_params=False):
        if not self.log_and_save:
            return
        name = 'IIR_LOG_%d' % self.log_idx

        if write_params:
            f = open(self.LOG_NAME, 'a') # append to the log

            f.write('\n*** %s ***\n' % name)
            f.write('Time=%s\n' % dt.now())
            f.write('Host=%s\n' % socket.gethostname())

            f.write("patch_dim =%d\n" % self.patch_dim)
            f.write("neurons   =%d\n" % self.neurons)
            f.write("cells     =%d\n" % self.cells)

            f.write("batch_size =%d\n" % self.batch_size)
            f.write("time_batch_size =%d\n" % self.time_batch_size)
            f.write("M_backprop   =%f\n" % self.M_backprop_steps  )

            f.write("Phi_eta_init =%f\n" % self.Phi_eta_init)
            f.write("M_eta_init   =%f\n" % self.M_eta_init  )
            f.write("B_eta_init   =%f\n" % self.B_eta_init  )

            f.write("eta_inc  =%d\n" % self.eta_inc )

            f.write("Phi_norm =%f\n" % self.Phi_norm)
            f.write("M_norm   =%f\n" % self.M_norm  )
            f.write("B_norm   =%f\n" % self.B_norm  )

            f.write("citers    =%d\n" % self.citers   )
            f.write("coeff_eta =%f\n" % self.coeff_eta)
            f.write("lambdav   =%f\n" % self.lambdav  )

            f.write("data_name =%s\n" % self.data_name)
            f.write("visualizer =%s\n" % self.visualizer)
            f.write('%d\n' % (self.log_idx))

            f.close()

            path = 'iir_dict/%s' % name
            if not os.path.exists(path):
                os.makedirs(path)

            logfile = '%s/%s_log.txt' % (path, name)
            print 'Assigning stdout to %s' % logfile
            sys.stdout = open(logfile, 'w') # Rewire stdout to write the log

        np.savez('iir_dict/%s' % name, Phi=Phi, B=B, M=M)

    def get_log_idx(self):
        f = open(self.LOG_NAME, 'r')
        rr = 0
        while True:
            r = f.readline()
            if r == '':
                break
            rr = r
        f.close()
        return int(rr) + 1

    def __init__(self):
        self.images = scipy.io.loadmat('mat/%s.mat' % self.data_name)
        self.images = self.images[self.data_name]
        (self.imsize, imsize, self.num_images) = np.shape(self.images)
        self.patch_per_dim = int(np.floor(imsize / self.sz))
        plt.ion()
        self.log_idx = self.get_log_idx()

    def load_videos(self):
        imsize, sz, tbs = self.imsize, self.sz, self.time_batch_size
        VI = np.zeros((self.patch_dim, self.time_batch_size, self.batch_size))
        for x in range(self.batch_size):
            # Choose a random image less than time_batch_size images
            # away from the end
            imi = np.floor((self.num_images - tbs) * random.uniform(0, 1))
            r = (imsize-sz) * random.uniform(0, 1)
            c = (imsize-sz) * random.uniform(0, 1)
            if self.coeff_test:
                imi, r, c = (0,0,0)
            img =  self.images[r:r+sz, c:c+sz, imi:imi+tbs]
            VI[:,:,x] = np.reshape(img, (self.patch_dim, tbs), 1)
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

    def profile_print(self, msg, start, override=False):
        if not self.profile and not override:
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

    def draw_potential(self, c, v, recon, VI):
        if not self.graphics_initialized:
            self.graphics_initialized = True
            fg, self.ax = plt.subplots(2,2)

            self.ax[1,1].set_title('Coefficients T=%d' % c)
            self.ax[1,1].set_xlabel('Coefficient Index')
            self.ax[1,1].set_ylabel('Activity')


            self.coeffs = self.ax[1,1].bar(range(self.cells), np.abs(v), color='r', lw=0)
            self.lthresh = self.ax[1,1].plot(range(self.cells+1), [self.lambdav] * (self.cells+1), color='g')

            axis_height = 2.05 # self.lambdav * 5
            self.ax[1,1].axis([0, self.cells, 0, axis_height])

            '''
            recon = t2dot(Phi, a)
            self.ax[0,0].imshow(np.reshape(recon, (self.sz, self.sz)),cmap = cm.binary, interpolation='nearest')
            self.ax[0,0].set_title('Iter=%d\nReconstruct' % 0)

            self.ax[0,1].set_title('Reconstruction Error')
            self.ax[0,1].set_xlabel('Time (steps)')
            self.ax[0,1].set_ylabel('SNR (dB)')
            self.ax[0,1].axis([0, self.num_coeff_upd, 0.0, 40])

            # The subplots move around if I don't do this lol...
            for i in range(6):
                plt.savefig('animation/junk.png')
            if self.save_cgraphs:
                plt.savefig('animation/%d.jpeg' % self.iter_idx)
            self.iter_idx += 1
            '''

        for coeff, i in zip(self.coeffs, range(self.cells)):
            coeff.set_height(abs(v[i]))  # Update the potentials

        time.sleep(0.1)

        '''
        #self.ax[1,0].imshow(np.reshape(I[:,0], (self.sz, self.sz)),cmap = cm.binary, interpolation='nearest')
        self.ax[1,0].set_title('Image')

        # Update Reconstruction
        recon = t2dot(Phi, a)
        self.ax[0,0].imshow(np.reshape(recon, (self.sz, self.sz)),cmap = cm.binary, interpolation='nearest')
        #self.ax[0,0].set_title('Iter=%d\nReconstruct' % self.iter_idx)
        self.ax[0,0].set_title('Iter=%d+%.2f\nReconstruct' % (self.iter_idx, random.random()))

        # Plot SNR
        var = I.var().mean()
        R = I - recon
        mse = (R ** 2).mean()
        snr = 10 * log(var/mse, 10)
        color = 'r'
        self.ax[0,1].scatter(self.iter_idx, snr, s=8, c=color)
        self.iter_idx += 1 # XXX Need to fix savfig + iter_idx
        '''

        plt.draw()
        plt.show()

    def draw(self, c, u, a, recon, VI):
        fg, ax = plt.subplots(4)

        ax[2].set_title('A Activity')
        ax[3].set_title('U Activity')
        a_height = np.max(np.abs(a))
        u_height = np.max(np.abs(u))
        ax[2].axis([0, self.time_batch_size, -a_height, a_height])
        ax[3].axis([0, self.time_batch_size, -u_height, u_height])
        for i in range(self.cells):
            ax[2].plot(range(self.time_batch_size), a[:,i,0], label='Coeff=%d' % i)
            ax[3].plot(range(self.time_batch_size), u[:,i,0], label='Coeff=%d' % i)
        ax[2].legend(bbox_to_anchor=(0.8, 0.6), loc=2, fontsize=10)
        ax[3].legend(bbox_to_anchor=(0.8, 0.6), loc=2, fontsize=10)

        for t in range(self.time_batch_size):
            ax[0].imshow(np.reshape(recon[:,t,0], (self.sz, self.sz)), cmap = cm.binary, interpolation='nearest')
            ax[0].set_title('Recon iter=%d, t=%d' % (c, t))
            ax[1].imshow(np.reshape(VI[:,t,0], (self.sz, self.sz)), cmap = cm.binary, interpolation='nearest')
            ax[1].set_title('Image iter=%d, t=%d' % (c, t))

            plt.draw()
            time.sleep(0.2)
        time.sleep(9999)
        plt.close()

    def grad_M(self, VI, Phi, M, B, error, u, a):
        start = dt.now()

        I = np.eye(self.neurons)
        iplusm = np.zeros((self.neurons, self.neurons, self.time_batch_size))
        iplusm[:,:,0] = I
        for t in range(1, self.time_batch_size):
            iplusm[:,:,t] = t2dot(iplusm[:,:,t-1], I+M)

        check_num_grad = False

        steps = self.M_backprop_steps
        result = np.zeros((self.neurons,self.neurons))
        tmp = np.zeros((self.neurons,self.neurons))
        for t in range(1, self.time_batch_size):
            # pb, pl, nb -> ln (before)

            # pb, pl -> bl
            x = tdot(error[:,t,:], Phi, [[0], [0]])

            for j in range(t-1, max(-1, t-1-steps), -1):
                # nn, nb -> nb OR ln, nb -> lb?
                y = tdot(iplusm[:,:,t-j-1], u[j,:,:], [[1], [0]])
                result += tdot(x, y, [[0], [1]])

            if check_num_grad and t == 1:
                E = 0.5 * np.linalg.norm(error[:,:,t]) ** 2
                e = 0.0001 # epsilon
                M_e = np.copy(M)
                M_e[0,0] += e
                u_e, recon_e = self.get_reconstruction(VI, Phi, M_e, B, a)
                error_e = VI - recon_e
                E_e = 0.5 * np.linalg.norm(error_e[:,:,t]) ** 2

                num_grad = - (E_e - E)/e

                print 'grad:     ', result[0,0]
                print 'num_grad: ', num_grad


        self.profile_print("dM Calc", start)
        return result

    def grad_Phi(self, error, u):
        start = dt.now()
        result = tdot(error, u, [[1,2], [0,2]]) # ptb * tnb
        self.profile_print("dPhi Calc", start)
        return result

    # Implement backprop?
    def grad_B(self, error, Phi, a):
        return np.zeros((self.cells, self.neurons))
        start = dt.now()
        x = tdot(error[:,1:,:], Phi, [[0], [0]]) # ptb * pn -> tbn
        result = tdot(x, a[:-1,:,:], [[0,1], [0,2]]) # tbn * tjb -> nj
        self.profile_print("dB Calc", start)
        return result

    def get_u(self, M, B, a):
        u = np.zeros((self.time_batch_size, self.neurons, self.batch_size))
        iplusm = np.eye(self.neurons) + M
        dot = t2dot
        u[0,:,:] = dot(B, a[0,:,:])
        for t in range(1, self.time_batch_size):
            u[t,:,:] = dot(iplusm, u[t-1,:,:]) + dot(B, a[t,:,:])
        return u

    def get_reconstruction(self, VI, Phi, M, B, a):
        start = dt.now()
        u = self.get_u(M, B, a)
        #self.profile_print("get_u Calc", start)
        r = np.tensordot(Phi, u, [[1], [1]])
        self.profile_print("get_reconstruction Calc", start)
        return u, r

    def check_activity(self, a):
        if np.sum(np.abs(a)) > (self.cells * self.time_batch_size * self.batch_size * 0.50): # Coeff explosion check
            print 'Activity Explosion!!!'
            print 'Data:'
            #print 'a (sum, min, max)',np.sum(np.abs(a)), np.min(a), np.max(a)
            print 'a (min, max)', np.min(a), np.max(a)
            print 'a index', np.where(a==np.min(a)), np.where(a==np.max(a))
            return True
        return False

    def get_Phi_hat(self, Phi, M, B):
        start = dt.now()

        I        = np.eye(self.neurons)
        Phi_hat  = np.zeros((self.patch_dim, self.cells, self.time_batch_size))
        iplusm_t = I
        Phi_hat[:,:,0] = t3dot(Phi, iplusm_t, B) # pn * nn * nc
        for t in range(1, self.time_batch_size):
            iplusm_t = t2dot(iplusm_t, I+M)
            Phi_hat[:,:,t] = t3dot(Phi, iplusm_t, B)

        self.profile_print("phi hat Calc", start)
        return Phi_hat

    def get_b(self, VI, Phi, Phi_hat, steps=1000):
        start = dt.now()
        b = np.zeros((self.time_batch_size, self.cells, self.batch_size))
        for t in range(self.time_batch_size):
            for T in range(t, min(self.time_batch_size, t+steps)):
                b[t,:,:] += t2dot(Phi_hat[:,:,T-t].T, VI[:,T,:]) # JP * PB = JB
        self.profile_print("b", start)
        return b

    def print_status(self, c, VI, error, a, u):
        print '\t%d) SNR=%.2fdB, E=%.3f A_Activity=%.2f%% U_Activity=%.2f%%' % \
            (c, self.get_snr(VI, error), np.sum(np.abs(error)),
             self.get_activity(a), self.get_activity(u))

    def sparsify(self, VI, Phi, M, B, debug=False):
        a = np.zeros((self.time_batch_size, self.cells, self.batch_size))
        #a[0,0,:] = 1
        u, recon = self.get_reconstruction(VI, Phi, M, B, a)
        error = VI - recon
        self.print_status(-1, VI, error, a, u)

        Phi_hat = self.get_Phi_hat(Phi, M, B)
        b = self.get_b(VI, Phi, Phi_hat)

        start = dt.now()
        for c in range(self.citers):
            Gu = np.zeros((self.time_batch_size, self.cells, self.batch_size))

            xstart = dt.now()
            steps = 30
            for t in range(self.time_batch_size - 1):
                end1 = min(steps, self.time_batch_size-t-1)
                t_phi_hat = Phi_hat[:,:,:end1]
                end2 = min(t+1+steps, self.time_batch_size)
                Gu[t,:,:] += comp_Gu(t_phi_hat, recon[:, t+1:end2, :])
            self.profile_print("\titer Calc", xstart)

            da = b - Gu - self.lambdav * self.sparse_cost(a)
            a += self.coeff_eta * da

            u, recon = self.get_reconstruction(VI, Phi, M, B, a)
            error = VI - recon
            self.print_status(c, VI, error, a, u)

            if c == self.citers - 1 or c % (self.citers/4) == 0:
                if self.visualizer and c == self.citers - 1:
                    self.draw(c, u, a, recon, VI)
        self.profile_print("iters Calc", start)

        return error, recon, a

    def sparsify(self, VI, Phi, M, B, debug=False):
        a = np.zeros((self.time_batch_size, self.cells, self.batch_size))
        #a[0,0,:] = 1.0
        u, recon = self.get_reconstruction(VI, Phi, M, B, a)
        error = VI - recon
        self.print_status(-1, VI, error, a, u)

        Phi_hat = self.get_Phi_hat(Phi, M, B)
        steps = 4
        b = self.get_b(VI, Phi, Phi_hat, steps)

        start = dt.now()
        for t in range(self.time_batch_size):
            print t
            for c in range(self.citers):
                end1 = min(steps, self.time_batch_size-t)
                t_phi_hat = Phi_hat[:,:,:end1]
                end2 = min(t+steps, self.time_batch_size)
                Gu = comp_Gu(t_phi_hat, recon[:, t:end2, :])

                da = b[t,:,:] - Gu - self.lambdav * steps * self.sparse_cost(a[t,:,:])
                a[t,:,:] += self.coeff_eta * da

                u, recon = self.get_reconstruction(VI, Phi, M, B, a)
                error = VI - recon
                self.print_status(c, VI, error, a, u)

                if c == self.citers - 1 or c % (self.citers/4) == 0:
                    if self.visualizer and c == self.citers - 1 and t == self.time_batch_size - 1:
                        self.draw(c, u, a, recon, VI)
        self.profile_print("iters Calc", start)

        return error, recon, a

    def sparsify(self, VI, Phi, M, B, debug=False):
        a = np.zeros((self.time_batch_size, self.cells, self.batch_size))
        #a[0,0,:] = 1.0
        u, recon = self.get_reconstruction(VI, Phi, M, B, a)
        error = VI - recon
        self.print_status(-1, VI, error, a, u)

        Phi_hat = self.get_Phi_hat(Phi, M, B)
        steps = 4
        b = self.get_b(VI, Phi, Phi_hat, steps)

        start = dt.now()
        for t in range(self.time_batch_size):
            print t
            for c in range(self.citers):
                end1 = min(steps, self.time_batch_size-t)
                t_phi_hat = Phi_hat[:,:,:end1]
                end2 = min(t+steps, self.time_batch_size)
                Gu = comp_Gu(t_phi_hat, recon[:, t:end2, :])

                da = b[t,:,:] - Gu - self.lambdav * steps * self.sparse_cost(a[t,:,:])
                a[t,:,:] += self.coeff_eta * da

                u, recon = self.get_reconstruction(VI, Phi, M, B, a)
                error = VI - recon
                self.print_status(c, VI, error, a, u)

                if c == self.citers - 1 or c % (self.citers/4) == 0:
                    #if self.visualizer and c == self.citers - 1 and t == self.time_batch_size - 1:
                    if self.visualizer and c == self.citers - 1:
                        self.draw(c, u, a, recon, VI)
        self.profile_print("iters Calc", start)

        return error, recon, a

    def thresh(self, v, theta):
        'LCA threshold function'
        if self.thresh_type=='hard': # L0 Approximation
            a = np.copy(v)
            a[np.abs(a) < theta] = 0
        elif self.thresh_type=='soft': # L1 Approximation
            a = abs(v) - theta;
            a[a < 0] = 0
            a = np.sign(v) * a
        return a

    def sparsify(self, VI, Phi, M, B, debug=False):
        v = np.zeros((self.time_batch_size, self.cells, self.batch_size))
        #v[0,0,:] = 1.0 + self.lambdav
        a = self.thresh(v, self.lambdav)

        u, recon = self.get_reconstruction(VI, Phi, M, B, a)
        error = VI - recon
        self.print_status(-1, VI, error, a, u)

        Phi_hat = self.get_Phi_hat(Phi, M, B)
        steps = 15
        b = self.get_b(VI, Phi, Phi_hat, steps)

        start = dt.now()
        for t in range(self.time_batch_size):
            print t
            for c in range(self.citers):
                end1 = min(steps, self.time_batch_size-t)
                t_phi_hat = Phi_hat[:,:,:end1]
                end2 = min(t+steps, self.time_batch_size)
                Gu = comp_Gu(t_phi_hat, recon[:, t:end2, :])

                #print 'b', b[t,:,:]
                #print 'Gu', Gu
                #print 'v', v[t,:,:]
                #print 'a', a[t,:,:]
                #pdb.set_trace()

                v[t,:,:] = self.coeff_eta * (b[t,:,:] - Gu) + (1 - self.coeff_eta) * v[t,:,:]
                #print np.max(v[t,:,:])
                a[t,:,:] = self.thresh(v[t,:,:], self.lambdav)

                u, recon = self.get_reconstruction(VI, Phi, M, B, a)
                error = VI - recon
                self.print_status(c, VI, error, a, u)

                #self.draw_potential(c, v[t,:,:], recon, VI)

                if c == self.citers - 1 or c % (self.citers/4) == 0:
                    #if self.visualizer and c == self.citers - 1 and t == self.time_batch_size - 1:
                    if self.visualizer and c == self.citers - 1:
                        self.draw(c, u, a, recon, VI)
        self.profile_print("iters Calc", start)

        return error, recon, a

    def norm_Phi(self, Phi):
        return t2dot(Phi, np.diag(self.Phi_norm/np.sqrt(np.sum(Phi**2, axis = 0))))

    def norm_B(self, B):
        return t2dot(B, np.diag(self.B_norm/np.sqrt(np.sum(B**2, axis = 0))))

    def norm_M(self, M):
        return t2dot(M, np.diag(self.M_norm/np.sqrt(np.sum(M**2, axis = 0))))

    def train(self):
        if self.coeff_test:
            name = 'Phi_399_0.4.mat'
            name = 'Phi_463/Phi_463_0.3.mat'
            Phi_orig = scipy.io.loadmat('dict/%s' % name)['Phi']

            #idx_0 = 53
            #idx_1 = 55
            idx_0 = 167
            idx_1 = 122
            Phi = np.zeros((Phi_orig.shape[0], 4))
            # Forward
            Phi[:,0] = Phi_orig[:,idx_0]
            Phi[:,2] = Phi_orig[:,idx_0]
            # Backward
            Phi[:,1] = Phi_orig[:,idx_1]
            Phi[:,3] = Phi_orig[:,idx_1]

            M = np.zeros((self.cells, self.cells))
            # Forward
            M[0][1] = -0.1
            M[1][0] = 0.1
            # Backward
            M[2][3] = 0.1
            M[3][2] = -0.1

            B = np.eye(self.cells)
        else:
            if self.load_phi is not None:
                d = np.load('iir_dict/IIR_LOG_%d.npz' % self.load_phi)
                Phi, B, M = d['Phi'], d['B'], d['M']
            else:
                Phi = np.random.randn(self.patch_dim, self.neurons)
                Phi = self.norm_Phi(Phi)

                B = 0.1 * np.random.randn(self.neurons, self.cells)
                B = self.norm_B(B)
                #B = np.eye(self.cells)
                M = 0.1 * np.random.randn(self.neurons, self.neurons)
                M = self.norm_M(M)

        if self.visualizer:
            self.batch_size = 1

        start = dt.now()
        for trial in range(self.num_trials):
            VI = self.load_videos()
            error, recon, a = self.sparsify(VI, Phi, M, B)

            print '%d) 1-SPARSIFY-SNR=%.2fdB' % (trial, self.get_snr(VI, error))
            u, recon = self.get_reconstruction(VI, Phi, M, B, a)

            if self.coeff_test:
                time.sleep(5)
                continue

            grad_M = self.grad_M(VI, Phi, M, B, error, u, a)
            grad_B = self.grad_B(error, Phi, a)
            grad_Phi = self.grad_Phi(error, u)

            M += self.get_eta(self.M_eta_init, trial) * grad_M
            u, recon = self.get_reconstruction(VI, Phi, M, B, a)
            error = VI - recon
            print '%d) 2-GRAD_M-SNR=%.2fdB' % (trial, self.get_snr(VI, error))

            B += self.get_eta(self.B_eta_init, trial) * grad_B
            u, recon = self.get_reconstruction(VI, Phi, M, B, a)
            error = VI - recon
            print '%d) 3-GRAD_B-SNR=%.2fdB' % (trial, self.get_snr(VI, error))

            Phi += self.get_eta(self.Phi_eta_init, trial) * grad_Phi
            u, recon = self.get_reconstruction(VI, Phi, M, B, a)
            error = VI - recon
            print '%d) 4-GRAD_Phi-SNR=%.2fdB' % (trial, self.get_snr(VI, error))

            #B = self.norm_B(B)
            Phi = self.norm_Phi(Phi)

            print "t=%d, Elapsed=%d seconds" % (trial, (dt.now()-start).seconds)
            self.log(Phi, B, M, trial==0)
            sys.stdout.flush()

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
