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
    patch_dim = 64
    neurons   = patch_dim * 4
    cells     = patch_dim * 1

    load_phi = 3
    log_and_save = False
    batch_size = 20
    time_batch_size = 70
    #batch_size = 10
    #time_batch_size = 64
    num_trials = 10000

    M_backprop_steps = 20

    Phi_eta_init = 1.2  / time_batch_size
    M_eta_init   = (0.02 / (M_backprop_steps)) / time_batch_size # Depend on the dim on input?
    B_eta_init   = 0.02 / time_batch_size # Depend on the dim on input?

    eta_inc  = 500

    Phi_norm = 1
    M_norm   = 0.01 # Not running except for on init
    B_norm   = 0.1

    citers    = 6
    coeff_eta = 5e-4
    lambdav   = 0.20 * time_batch_size # (have to account for sum over time)

    data_name = 'IMAGES_DUCK_SHORT'
    profile = False
    visualizer = False
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
        for i in range(self.cells):
            ax[2].plot(range(self.time_batch_size), a[i,0,:])

        for t in range(self.time_batch_size):
            ax[0].imshow(np.reshape(recon[:,0,t], (self.sz, self.sz)), cmap = cm.binary, interpolation='nearest')
            ax[0].set_title('Recon iter=%d, t=%d' % (c, t))
            ax[1].imshow(np.reshape(VI[:,0,t], (self.sz, self.sz)), cmap = cm.binary, interpolation='nearest')
            ax[1].set_title('Image iter=%d, t=%d' % (c, t))

            plt.draw()
            time.sleep(0.02)
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
        start = dt.now()
        x = tdot(error[:,1:,:], Phi, [[0], [0]]) # ptb * pn -> tbn
        result = tdot(x, a[:-1,:,:], [[0,1], [0,2]]) # tbn * tjb -> nj
        self.profile_print("dB Calc", start)
        return result

    def get_u(self, M, B, a):
        u = np.zeros((self.time_batch_size, self.neurons, self.batch_size))
        iplusm = np.eye(self.neurons) + M
        u[0,:,:] = np.dot(B, a[0,:,:])
        for t in range(1, self.time_batch_size):
            u[t,:,:] = np.dot(iplusm, u[t-1,:,:]) + np.dot(B, a[t,:,:])
        return u

    def get_reconstruction(self, VI, Phi, M, B, a):
        start = dt.now()
        u = self.get_u(M, B, a)
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

    def get_b_and_G(self, VI, Phi, Phi_hat):
        start = dt.now()
        tbs = self.time_batch_size
        b = np.zeros((self.time_batch_size, self.cells, self.batch_size))
        G = np.zeros((self.time_batch_size, self.cells, self.neurons))
        for t in range(self.time_batch_size):
            for T in range(t+1, self.time_batch_size):
                b[t,:,:] += np.dot(Phi_hat[:,:,T-(t+1)].T, VI[:,T,:]) # JP * PB = JB
                #G[t,:,:] += np.dot(Phi_hat[:,:,T-(t+1)].T, Phi)
            G[t,:,:] = np.dot(Phi_hat[:,:,t].T, Phi)
        #for t in range(self.time_batch_size-2, -1, -1):
            #for T in range(t+1, self.time_batch_size):
                #b[t,:,:] += np.dot(Phi_hat[:,:,T-(t+1)].T, VI[:,T,:]) # JP * PB = JB
            #G[t,:,:] = G[t+1,:,:] + np.dot(Phi_hat[:,:,tbs-t-2].T, Phi)

        self.profile_print("b and G Calc", start)
        return b, G

    def print_status(self, c, VI, error, a, u):
        print '\t%d) SNR=%.2fdB, E=%.3f A_Activity=%.2f%% U_Activity=%.2f%%' % \
            (c, self.get_snr(VI, error), np.sum(np.abs(error)),
             self.get_activity(a), self.get_activity(u))

    def sparsify(self, VI, Phi, M, B, debug=False):
        a = np.zeros((self.time_batch_size, self.cells, self.batch_size))
        #a = 0.1 * np.random.randn(self.time_batch_size, self.cells, self.batch_size)
        u, recon = self.get_reconstruction(VI, Phi, M, B, a)
        error = VI - recon

        self.print_status(-1, VI, error, a, u)
        start_snr = self.get_snr(VI, error)
        last_snr = None

        Phi_hat = self.get_Phi_hat(Phi, M, B)
        b,G = self.get_b_and_G(VI, Phi, Phi_hat)

        I = np.eye(self.neurons)
        iplusm = np.zeros((self.neurons, self.neurons, self.time_batch_size))
        iplusm[:,:,0] = I
        for t in range(1, self.time_batch_size):
            iplusm[:,:,t] = t2dot(iplusm[:,:,t-1], I+M)

        start = dt.now()
        for c in range(self.citers):
            #print 'U average', np.average(np.abs(u))
            da = np.zeros((self.time_batch_size, self.cells, self.batch_size))
            gg = b
            da += b
            print 'Diff', np.max(da - gg)
            pdb.set_trace()
            for t in range(self.time_batch_size):
                for T in range(t+1, self.time_batch_size):
                    #da[t,:,:] -= np.dot(Phi_hat[:,:,T-t-1].T, np.dot(Phi, u[T,:,:]))
                    r = np.dot(Phi, u[T,:,:])
                    rp = np.dot(r.T, Phi)
                    rpi = np.dot(rp, iplusm[:,:,T-t-1])
                    rpib = np.dot(rpi, B)
                    da[t,:,:] -= rpib.T
                if c == 2 and t == self.time_batch_size - 2:
                    T = t+1
                    #  pb * pn -> bn * nj -> bj .T -> jb
                    x = np.dot(tdot(error[:,T,:], Phi, [[0], [0]]), B).T

            #da += b
            da -= self.lambdav * self.sparse_cost(a)
            print 'DA average', np.average(np.abs(da[self.time_batch_size-2,:,:]))
            #da = b - t_bdot(G,u) - self.lambdav * self.sparse_cost(a)
            a += self.coeff_eta * da
            #print 'A average', np.average(np.abs(a))


            u, recon = self.get_reconstruction(VI, Phi, M, B, a)
            error = VI - recon
            #print 'error average', np.average(np.abs(error))

            #if c == self.citers - 1 or c % (self.citers/4) == 0:
            if c < 5:
                self.print_status(c, VI, error, a, u)
                last_snr = self.get_snr(VI, error)
                if self.visualizer and c == self.citers - 1:
                    self.draw(c, a, recon, VI)
        self.profile_print("iters Calc", start)

        return error, recon, a

    def get_u_2(self, M, B, a):
        u = np.zeros((self.neurons, self.batch_size,self.time_batch_size))
        iplusm = np.eye(self.neurons) + M
        u[:,:,0] = np.dot(B, a[:,:,0])
        for t in range(1, self.time_batch_size):
            u[:,:,t] = np.dot(iplusm, u[:, :, t-1]) + np.dot(B, a[:,:,t])
        return u

    def get_reconstruction_2(self, VI, Phi, M, B, a):
        start = dt.now()
        u = self.get_u_2(M, B, a)
        r = np.tensordot(Phi, u, [[1], [0]])
        self.profile_print("get_reconstruction Calc", start)
        return u, r

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

    def grad_a(self, VI, error, Phi, M, B, debug=True):
        I = np.eye(self.neurons)
        iplusm = np.zeros((self.neurons, self.neurons, self.time_batch_size))
        iplusm[:,:,0] = I
        for t in range(1, self.time_batch_size):
            iplusm[:,:,t] = np.dot(iplusm[:,:,t-1], I+M)

        start = dt.now()
        result = np.zeros((self.cells, self.batch_size, self.time_batch_size))
        r2     = np.zeros((self.cells, self.batch_size, self.time_batch_size))
        steps = 1000
        for TT in range(self.time_batch_size - 1):
            end1 = min(steps, self.time_batch_size-TT-1)
            tmp = iplusm[:,:,:end1]

            end2 = min(self.time_batch_size, TT+1+steps)
            result[:,:,TT] = grad_a_TT(error[:,:,TT+1:end2], Phi, tmp, B)

            #y = np.tensordot(tmp, B, [[1], [0]]) # lnt nj -> ltj
            #z = np.tensordot(Phi, y, [[1], [0]]) # pl ltj -> ptj
            #r2[:,:,TT] = np.tensordot(z, error[:,:,TT+1:end2], [[0,1], [0,2]]) # ptj pbt -> jb

        self.profile_print("dA Calc", start)

        if debug:
            r2 = self.debug_a(error, Phi, M, B, iplusm)
            print np.max(result - r2)
            assert np.allclose(result, r2)

        return result

    def sparsifya(self, VI, Phi, M, B, debug=False):
        VI = np.swapaxes(VI, 1, 2)
        #a = np.zeros((self.cells, self.batch_size, self.time_batch_size))
        a = np.zeros((self.cells, self.batch_size, self.time_batch_size))
        #a = 1.0 * np.random.randn(self.cells, self.batch_size, self.time_batch_size)
        u, recon = self.get_reconstruction_2(VI, Phi, M, B, a)
        error = VI - recon

        print '\t%d) SNR=%.2fdB, E=%.3f Activity=%.2f%%' % \
            (-1, self.get_snr(VI, error), np.sum(np.abs(error)),
             self.get_activity(a))

        for c in range(self.citers):
            #print 'U average', np.average(np.abs(u))
            da = self.grad_a(VI, error, Phi, M, B, debug) - \
                        self.lambdav * self.sparse_cost(a)
            print 'DA average', np.average(np.abs(da[:,:,self.time_batch_size-2]))
            a += self.coeff_eta * da
            #print 'A average', np.average(np.abs(a))

            u, recon = self.get_reconstruction_2(VI, Phi, M, B, a)
            error = VI - recon

            #print 'error average', np.average(np.abs(error))
            #if c < 5:
            if c == self.citers - 1 or c % (self.citers/4) == 0:
                print '\t%d) SNR=%.2fdB, E=%.3f Activity=%.2f%%' % \
                    (c, self.get_snr(VI, error), np.sum(np.abs(error)),
                     self.get_activity(a))
                if self.visualizer and c == self.citers - 1:
                #if self.visualizer:
                    self.draw(c, a, recon, VI)

        return error, recon, a


    def norm_Phi(self, Phi):
        return np.dot(Phi, np.diag(self.Phi_norm/np.sqrt(np.sum(Phi**2, axis = 0))))

    def norm_B(self, B):
        return np.dot(B, np.diag(self.B_norm/np.sqrt(np.sum(B**2, axis = 0))))

    def norm_M(self, M):
        return np.dot(M, np.diag(self.M_norm/np.sqrt(np.sum(M**2, axis = 0))))

    def train(self):
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
            print 'New'
            error, recon, a = self.sparsify(VI, Phi, M, B)
            print 'Orig'
            error, recon, a = self.sparsifya(VI, Phi, M, B)

            print '%d) 1-SPARSIFY-SNR=%.2fdB' % (trial, self.get_snr(VI, error))
            u, recon = self.get_reconstruction(VI, Phi, M, B, a)

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

            B = self.norm_B(B)
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
