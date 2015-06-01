import matplotlib
matplotlib.rcParams.update({'figure.autolayout': True}) # Magical tight layout

import traceback
import math
import socket
import os
import scipy.io
import scipy.stats as stats
from datetime import datetime
import time
import numpy as np
from numpy import reshape, zeros, ones
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from math import floor, ceil, sqrt
import pprint
import argparse
import pdb
from helpers import *
from Recurrnet.recurrnet.optimizer import *
from multiprocessing import Pool, Queue, Manager
import multiprocessing
import sys

dtype = theano.config.floatX
class Analysis():

    LOG_NAME = 'aalog.txt'
    datasets = [
                #'IMAGES_DUCK'
                ['IMAGES_DUCK_SHORT', 0, 15],#
                #['IMAGES_EDGE_DUCK',           0, 100],
                #['IMAGES_EDGE_DUCK', 0, 100],
                #['IMAGES_EDGE_DUCK_r=20_c=20', 0, 100],
                #['IMAGES_EDGE_DUCK_r=21_c=21', 0, 100],
                #['IMAGES_EDGE_DUCK_r=22_c=22', 0, 100],
                #['IMAGES_EDGE_DUCK_r=23_c=23', 0, 100],
                #['IMAGES_EDGE_DUCK_r=24_c=24', 0, 100],
               ]
    start_patch = 189
    patches = 1
    reconstruct_i = 2 # Which dataset index to reconstruct (in over_time())
    reset_after = 15
    cells = 10  # Number of Gamma's
    clambdav = 0.50
    citers  = 10
    normalize_Gamma = True
    #reset_after = -1 # -1 means never
    inertia = True
    sparsify = 1
    zero_diagonal = False
    profile = False

    a_mode  = True # ahat or u mode
    patch_i   = 189
    phi_name  = 'Phi_463_0.3'
    coeffs    = None

    # Training parameters
    num_trials = 3000
    eta_init   = 1.2
    eta_inc    = 80

    ceta       = 0.05
    log_often  = 8
    Z_init     = ''
    G_init     = ''
    #G_init     = 'AA_Log_121'
    #Z_init     = 'AA_Log_7'

    coeff_visualizer = False
    graphics_initialized = False

    def __init__(self):
        parr = self.phi_name.split('_')
        direc = parr[0] + '_' + parr[1]
        self.phi = scipy.io.loadmat('dict/%s/%s' % (direc,self.phi_name))['Phi']

        self.sz = np.sqrt(self.phi.shape[0])
        self.neurons = self.phi.shape[1]

        if self.coeff_visualizer:
            self.patches = 1

        'Log of ahat or u values'
        aname = 'activity' if self.a_mode else 'membrane'
        for ds in self.datasets:
            idn = ds[0]
            self.image = scipy.io.loadmat('mat/%s.mat' % idn)[idn]
            (self.imsize, imsize, self.num_images) = np.shape(self.image)
            self.patch_per_dim = int(np.floor(imsize / self.sz))

            self.log = np.load('activity/%s_%s_%s.npy' % (aname, self.phi_name, ds[0]))
            #self.log = self.log[:, self.start_patch:self.start_patch+self.patches, ds[1]:ds[2]]

        self.timepoints = min(self.log.shape[2], self.image.shape[2]) # num_frames
        self.batch_size = self.log.shape[1] # batch_size
        self.lambdav = 0.2

        self.log_idx = self.get_log_idx()

        if self.G_init != '':
            # Have to copy so Theano won't complain about unaligned np array
            self.Gam = np.copy(scipy.io.loadmat('activity/%s' % self.G_init)['Gam'])
        else:
            self.Gam = None

        if self.Z_init != '':
            self.Z = scipy.io.loadmat('activity/%s' % self.Z_init)['Z']
        else:
            self.Z = None

        if self.reset_after == -1:
            self.reset_after = 9e10 # Never

        np.set_printoptions(precision=4)
        #plt.ion()
        self.init_theano()

    def init_theano(self):
        if True:
            Ahat = T.fmatrix('Ahat')
            A_prev = T.fmatrix('A_prev')
            Gam = T.tensor3('Gam')
            C = T.fmatrix('C')
            E = 0.5 * ((Ahat - T.batched_dot(T.tensordot(Gam, C, 1).dimshuffle(2, 0, 1), A_prev.T).T).norm(2) ** 2)
            gc = T.grad(E, C)
            self.gc = function([Ahat, A_prev, Gam, C], gc, allow_input_downcast=True)

    def thresh(self, u, theta):
        'LCA threshold function'
        a = abs(u) - theta;
        a[a < 0] = 0
        a = np.sign(u) * a
        return a

    def find_coeffs(self, patch, tstart, tend):
        coeffs_per_frame = 1
        coeff_set = set([])
        coeffs = []
        for t in range(tstart, tend):
            activities = np.copy(np.abs(self.log[:, patch, t]))
            all_coeffs = np.fliplr([activities.argsort()])[0]
            good_coeffs = list(all_coeffs[0:coeffs_per_frame])

            for coeff in good_coeffs:
                if coeff in coeff_set:
                    continue
                coeffs.append(coeff)
            coeff_set = coeff_set.union(good_coeffs)

        return coeffs

    def power_over_time(self):
        'Display the norm of the coefficients over time'
        patch_i = self.patch_i
        for t in range(self.tstart, self.tend):
            print 't=%d) Norm=%f' % (t, np.linalg.norm(self.activity_log[:, patch_i, t]))

    def other(self):
        # Replace activity with predictions of data if desired
        if self.Z != None:
            for t in range(1, log.shape[1]):
                if t % self.reset_after == 0:
                    activity = log[:, t]
                    continue
                activity = np.dot(self.Z, activity)
                log[coeffs, t] = activity # overwrite actual activity

    def survey_activity(self):
        tlen = 20
        jobs = []
        for tstart in range(0, self.timepoints-tlen, tlen):
            for patch in range(self.batch_size):
                coeffs = self.find_coeffs(patch, tstart, tstart+tlen)
                jobs += [(patch, coeffs, tstart, tstart+tlen)]

        njobs = len(jobs)

        self.nprocesses = multiprocessing.cpu_count()

        start = datetime.now()
        m = Manager()
        q = m.Queue()
        p = Pool(self.nprocesses)
        running = 0

        for j in range(self.nprocesses):
            patch, coeffs, tstart, tend = jobs.pop()
            #p.apply_async(self.over_time, args=(patch, tstart, tstart+tlen, q))
            p.apply_async(test, args=(q, patch, tstart, tend))
            running += 1

        print running
        z = 0
        while running > 0:
            e = q.get()
            print e
            print 'Job %d/%d completed E=%d' % (z, njobs, (datetime.now() - start).seconds)
            sys.stdout.flush()
            z = z + 1

            if len(jobs):
                patch, coeffs, tstart, tend = jobs.pop()
                p.apply_async(over_time_2, args=(self.image, self.phi_name, self.patch_per_dim, self.sz, self.phi, self.log, coeffs, patch, tstart, tstart+tlen, q))
            else:
                running -= 1 # Didn't start another job to replace this one

        p.close()
        p.join()

    def over_time(self, patch, tstart, tend, q=None, time_only=False):
        'Plot the activity over time, the video, and the reconstruction, simultaneously'
        try:
            coeffs = self.coeffs
            if coeffs == None:
                coeffs = self.find_coeffs(patch, tstart, tend)
            print 'Plotting %d coeffs for p=%d, t=(%d,%d): %s' % (len(coeffs), patch, tstart, tend, coeffs)

            # Plot Setup
            log = self.log
            tlen = tend - tstart
            row_start = 2
            rows = 1 if time_only else 6
            cols = 6

            fg = matplotlib.pyplot.figure(figsize=(10.0, 10.0))

            # Time graph
            ax_time = plt.subplot2grid((rows,cols), (0,0), rowspan=row_start, colspan=cols)
            for i in coeffs:
                ax_time.plot(range(tlen), [0] * tlen, color='k') # X axis
                ax_time.plot([0, 0], [-1.0, 1.0], color='k')                 # Y axis

                ax_time.plot(range(tlen), log[i, patch, tstart:tend], label='A%d' % i)
                #ax_time.legend()
                #lg = ax_time.legend(bbox_to_anchor=(-0.6 , 0.40), loc=2, fontsize=10)
                #lg = ax_time.legend(bbox_to_anchor=(1.05, 1), loc=2, fontsize=50)
                lg = ax_time.legend(bbox_to_anchor=(0., 0, 1., .102), loc=3,
                           ncol=2, fontsize=10, borderaxespad=0.)
                lg.draw_frame(False)
                ax_time.axis([0, tlen - 1, -1, 1])
                ax_time.set_title("Patch=%d, t=(%d,%d) Dict=%s" % (patch, tstart, tend, self.phi_name))

            if not time_only:
                # Coefficients
                i = 0
                for r in range(row_start+1, rows):
                    for c in range(cols):
                        if i >= len(coeffs):
                            break
                        ax = plt.subplot2grid((rows,cols), (r,c))
                        ax.get_xaxis().set_visible(False)
                        ax.get_yaxis().set_visible(False)
                        ax.imshow(np.reshape(self.phi[:,coeffs[i]], (self.sz, self.sz)), cmap = cm.binary, interpolation='nearest')
                        ax.set_title("A%d" % coeffs[i])
                        i = i + 1

                # Reconstruction and Images
                direc = 'activity/images/'
                name = '%s_P%d_T%d.%d' % (self.phi_name, patch, tstart, tend)
                for iters in range(1): # Show forever
                    for t in range(tstart, tend):
                        #for log in self.logs:
                        k = 0
                        col = int(np.floor(cols/2) - 1)
                        ax_r = plt.subplot2grid((rows,cols), (row_start, col)) # Reconstruct
                        img = np.dot(self.phi[:, coeffs], log[coeffs,patch,t])

                        ax_r.set_title("Rec t=%d" % t)
                        ax_r.get_xaxis().set_visible(False)
                        ax_r.get_yaxis().set_visible(False)

                        ax_r.imshow(np.reshape(img, (self.sz, self.sz)), cmap = cm.binary, interpolation='nearest')

                        col = int(np.floor((cols/2) + 1) - 1)
                        ax_i = plt.subplot2grid((rows,cols), (row_start, col)) # Images
                        ax_i.get_xaxis().set_visible(False)
                        ax_i.get_yaxis().set_visible(False)
                        ax_i.set_title("Img  t=%d" % t)

                        rr = (np.floor(patch / self.patch_per_dim)) * self.sz
                        cc = (patch % self.patch_per_dim) * self.sz

                        dimg = self.image[rr:rr+self.sz, cc:cc+self.sz, t].T
                        ax_i.imshow(dimg, cmap = cm.binary, interpolation='nearest')
                        plt.draw()
                        #plt.savefig('%sstage/%s_t=%3d.png' % (direc, name, t))
                        plt.savefig('%s/%s_t=%3d.png' % (direc, name, t))
                        #mng = plt.get_current_fig_manager()
                        #mng.frame.Maximize(True)
                        #plt.show()

                        #dimg = self.images[k][rr:rr+self.sz, cc:cc+self.sz, t].T
                        #dimg = self.images[self.reconstruct_i][rr:rr+self.sz, cc:cc+self.sz, t].T
                        #plt.show(block=True)

            #os.system('convert -delay 40 -loop 0 %sstage/*.png activity/gifs/%s.gif' % (direc, name))
            os.system('convert -delay 40 -loop 0 %s/%s*.png activity/gifs/%s.gif' % (direc, name, name))
            #os.system('mv %sstage/*.png %s' % (direc, direc))
            plt.close()

            if q is not None:
                q.put(1)
        except Exception as e:
            if q is not None:
                q.put(e)
            else:
                raise e
        #plt.show(block=True)

    def spatial_correlation(self, log=False):
        'Make a scatter plot of the activity of a pair of coefficients'
        if len(coeffs) != 2:
            raise Exception("Greater than 2 dim not supported.")

        patches = range(self.batch_size)[:3]
        x = self.activity_log[coeffs[0], patches, :]
        y = self.activity_log[coeffs[1], patches, :]
        if log:
            x = np.log(np.abs(x))
            y = np.log(np.abs(y))
            h = 10
        else:
            x = np.abs(x)
            y = np.abs(y)
            h = max(np.max(np.abs(x)), np.max(np.abs(y)))

        print 'h', h
        plt.plot([-h, h], [0, 0], color='k')
        plt.plot([0, 0], [-h, h], color='k')

        plt.scatter(x, y, s=1, color='red')
        plt.xlabel("A%d" % coeffs[0], fontdict={'fontsize':18})
        plt.ylabel("A%d" % coeffs[1], fontdict={'fontsize':18})
        plt.axis('equal')
        plt.show()

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

    def log_and_save(self, varname, var, write_params=False):
        name = 'AA_Log_%d' % self.log_idx
        if write_params:
            f = open(self.LOG_NAME, 'a') # append to the log
            f.write('\n*** %s ***\n' % name)
            f.write('Time=%s\n' % datetime.now())
            f.write('Host=%s\n' % socket.gethostname())
            f.write('datasets=%s\n' % self.datasets)
            f.write('start_patch=%d\n' % self.start_patch)
            f.write('patches=%d\n' % self.patches)
            f.write('sparsify=%s\n' % self.sparsify)
            f.write('phi_name=%s\n' % self.phi_name)
            f.write('coeffs=%s\n' % self.coeffs)
            f.write('Z_init=%s\n' % self.Z_init)
            f.write('G_init=%s\n' % self.G_init)
            f.write('Norm_G=%s\n' % self.normalize_Gamma)
            f.write('Inertia=%s\n' % self.inertia)
            f.write('cells=%d\n' % self.cells)
            f.write('ceta=%f\n' % self.ceta)
            f.write('citersceta=%f\n' % self.citers)
            f.write('num_trials=%s\n' % self.num_trials)
            f.write('eta_init=%s\n' % self.eta_init)
            f.write('eta_inc=%s\n' % self.eta_inc)
            f.write('%d\n' % (self.log_idx))
            f.close()
        scipy.io.savemat('activity/%s' % name, {varname: var})

    def get_eta(self, t):
        'Return eta value for learning at time step t'
        eta = self.eta_init
        for i in range(1, 11):
            if t < i * self.eta_inc:
                return eta/self.batch_size
            eta /= 2.0
        return eta/self.batch_size

    def rerr(self, R):
        return np.sum(np.abs(R)) / (self.batch_size)

    def norm_Gam(self, Gam):
        if not self.normalize_Gamma:
            return Gam

        for i in range(self.cells):
            Gam[:,:,i] = t2dot(Gam[:,:,i], np.diag(1/np.sqrt(np.sum(Gam[:,:,i]**2, axis = 0))))
            #Gam[:,:,i] *= 1/np.sqrt(np.sum(Gam[:,:,i]**2))
            #Gam[:,:,i] *= 1/np.linalg.norm(Gam[:,:,i], 'fro')
        return Gam

    def profile_print(self, msg, time, ind=0):
        if not self.profile or ind != 0:
            return
        print '%s %7d microseconds' % (msg, time)

    def check_activity(self, c):
        if np.sum(np.abs(c)) > (self.cells * self.batch_size * 0.50): # Coeff explosion check
            print 'Activity Explosion!!!'
            print 'Data:'
            #print 'c (sum, min, max)',np.sum(np.abs(c)), np.min(c), np.max(c)
            print 'c (min, max)', np.min(c), np.max(c)
            print 'c index', np.where(c==np.min(c)), np.where(c==np.max(c))
            return True

    def calc_b(self, Gam, x_prev, x):
        #options = [0,1,2]
        options = [0]
        if 0 in options:
            start = datetime.now()
            b = csparsify_grad(x, Gam, x_prev).T
            self.profile_print('B0:', (datetime.now() - start).microseconds)
        if 1 in options:
            start = datetime.now()
            b = np.einsum('pB,pnT,nB->TB', x, Gam, x_prev)
            self.profile_print('B1:', (datetime.now() - start).microseconds)
        if 2 in options:
            start = datetime.now()
            tmp = np.einsum('pB,nB->pnB', x, x_prev)
            b = csparsify_grad_2(Gam, tmp)
            self.profile_print('B2:', (datetime.now() - start).microseconds)
        return b

    def c_grad(self, Gam, x_prev, x, c_prev=None):
        c = c_prev if c_prev is not None else np.zeros((self.cells, self.batch_size))

        b = self.calc_b(Gam, x_prev, x)
        #G2 = np.einsum('pri,pnT,rB,nB->BiT', Gam, Gam, x_prev, x_prev)
        start = datetime.now()
        G = G_LCA(Gam, Gam, x_prev, x_prev)
        self.profile_print('G0:', (datetime.now() - start).microseconds)

        for i in range(self.citers):
            gradient = (b - np.einsum('iB, BiT->TB', c, G)) # Optimize? 23 microseconds last time
            #gradient = -self.gc(x, x_prev, Gam, c)

            c += self.ceta * gradient
            self.check_activity(c)
            #print 'Activity: %.2f%%' % self.get_activity(c)
        return 0, c

    def csparsify_thresh(self, Gam, x_prev, x, v_prev=None):
        v = v_prev if v_prev is not None else np.zeros((self.cells, self.batch_size))
        c = self.thresh(v, self.clambdav)

        b = self.calc_b(Gam, x_prev, x)
        #G2 = np.einsum('pri,pnT,rB,nB->BiT', Gam, Gam, x_prev, x_prev)
        start = datetime.now()
        G = G_LCA(Gam, Gam, x_prev, x_prev)
        if self.zero_diagonal:
            for i in range(self.patches):
                np.fill_diagonal(G[i], 0)
        self.profile_print('G0:', (datetime.now() - start).microseconds)

        for i in range(self.citers):
            gradient = (b - np.einsum('iB, BiT->TB', c, G)) # Optimize? 23 microseconds last time
            #gradient = -self.gc(x, x_prev, Gam, c)
            v = self.ceta * gradient + (1 - self.ceta) * v
            c = self.thresh(v, self.clambdav)
            self.check_activity(c)
            #print 'Activity: %.2f%%' % self.get_activity(c)

        return v, c

    def csparsify(self, Gam, x_prev, x, c_prev=None):
        c = c_prev if c_prev is not None else np.zeros((self.cells, self.batch_size))

        b = self.calc_b(Gam, x_prev, x)
        #G2 = np.einsum('pri,pnT,rB,nB->BiT', Gam, Gam, x_prev, x_prev)
        start = datetime.now()
        G = G_LCA(Gam, Gam, x_prev, x_prev)
        self.profile_print('G0:', (datetime.now() - start).microseconds)

        for i in range(self.citers):
            sparse_cost = self.clambdav * (2 * c) / (c ** 2 + 1)
            gradient = (b - np.einsum('iB, BiT->TB', c, G)) - sparse_cost
            c += self.ceta * gradient
            self.check_activity(c)
            #print 'Activity: %.2f%%' % self.get_activity(c)

        return 0, c

    def get_activity(self, c):
        max_active = self.batch_size * self.cells
        chat_c = np.copy(c)
        chat_c[np.abs(chat_c) > self.clambdav/1000.0] = 1
        ac = 100 * np.sum(chat_c)/max_active
        return ac

    def train_G_dynamics(self):
        'Train a Z matrix to learn the dynamics of the coefficients'
        #Z = self.Z if self.Z is not None else np.zeros((self.neurons, self.neurons))
        if self.Gam is None:
            Gam = np.zeros((self.neurons, self.neurons, self.cells))

            for i in range(self.cells):
                Gam[:,:,i] = np.eye(self.neurons) + np.random.normal(0, 0.05, (self.neurons, self.neurons))
            #Gam[:,:,0] = np.eye(self.neurons) + np.random.normal(0, 0.01, (self.neurons, self.neurons))
            #for i in range(1, self.cells):
                #Gam[:,:,i] = np.random.normal(0, 0.25, (self.neurons, self.neurons))
            Gam = self.norm_Gam(Gam)
        else:
            Gam = self.Gam
        f = open('activity/logs/AA_%d_log.txt' % self.log_idx, 'w')

        alg_start = datetime.now()
        for k in range(self.num_trials):
            R_sum = np.zeros((self.neurons, self.batch_size))
            #C_sum = np.zeros((self.cells, self.batch_size))
            C_sum = 0

            c_prev = None
            v_prev = None
            for t in range(tstart+1, tend):
                # Data
                x_prev, x = (self.log[:, :, t-1], self.log[:, :, t])

                # Inference
                if self.sparsify == 0: # No sparsity
                    v, chat = self.c_grad(Gam, x_prev, x, c_prev=c_prev)
                elif self.sparsify == 1: # Threshold
                    v, chat = self.csparsify_thresh(Gam, x_prev, x, v_prev=v_prev)
                elif self.sparsify == 2: # log(1+x^2)
                    v, chat = self.csparsify(Gam, x_prev, x, c_prev=c_prev)

                # Residual
                #options = [0,1,2]
                options = [0]
                if 0 in options:
                    start = datetime.now()
                    tmp = np.einsum('tb,nb->bnt', chat, x_prev)
                    x_pred = tgam_predict(Gam, tmp)
                    self.profile_print('P0:', (datetime.now() - start).microseconds)
                if 1 in options: # 10x slower?
                    start = datetime.now()
                    x_pred = gam_predict(Gam, chat, x_prev).T
                    self.profile_print('P1:', (datetime.now() - start).microseconds)
                if 2 in options:
                    start = datetime.now()
                    x_pred = np.einsum('pnt,tb,nb->pb', Gam, chat, x_prev)
                    self.profile_print('P2:', (datetime.now() - start).microseconds)

                R = x - x_pred
                #print '\t\tR=%f' % (np.sum(np.abs(R)) / self.batch_size)

                #options = [0,1]
                options = [1] if have_gpu() else [0]
                if 0 in options:
                    start = datetime.now()
                    dGam = np.einsum('pb,tb,nb->pnt', R, chat, x_prev)
                    self.profile_print('D0:', (datetime.now() - start).microseconds)
                if 1 in options: # 2x slower, too much shuffling?
                    start = datetime.now()
                    dGam = t3tendot2(R, chat, x_prev)
                    self.profile_print('D1:', (datetime.now() - start).microseconds)
                elif False: # Really slow (Don't try it.)
                    start = datetime.now()
                    ttdGam = np.einsum('Pb,Nb,Tb->PNT', x, x_prev, chat) - np.einsum('ib,Tb,Pri,rb,Nb->PNT', chat, chat, Gam, x_prev, x_prev)
                    self.profile_print('D2:', (datetime.now() - start).microseconds)

                Gam += self.get_eta(k) * dGam
                Gam = self.norm_Gam(Gam)

                R_sum += np.abs(R)
                C_sum += self.get_activity(chat)
                if self.inertia:
                    v_prev = v
                    c_prev = chat

            e = (datetime.now() - alg_start).seconds
            r = np.sum(R_sum) / (self.timepoints * self.batch_size)
            msg = 'T=%.4d E=%ds, R=%f, C=%.2f%%' % (k, e, r, C_sum / self.timepoints)
            print msg
            f.write(msg +'\n')

            if k > 0  and k % self.log_often == 0:
                self.log_and_save('Gam', Gam, k == self.log_often)
                print 'Saved Gam_%d' % self.log_idx

        f.close()
        self.log_and_save('Gam', Gam)

    def train_dynamics(self):
        'Train a Z matrix to learn the dynamics of the coefficients'
        #Z = self.Z if self.Z is not None else np.zeros((self.neurons, self.neurons))
        Z = self.Z if self.Z is not None else np.eye(self.neurons)
        f = open('activity/logs/AA_%d_log.txt' % self.log_idx, 'w')

        #log(neurons, patches, timepoints)
        start = datetime.now()
        for k in range(self.num_trials):
            R_sum = np.zeros((self.neurons, self.batch_size))
            for t in range(1, self.timepoints):
                x_prev, x = (self.log[:, :, t-1], self.log[:, :, t])
                R = x - np.dot(Z, x_prev)
                Z = Z + self.get_eta(k) * np.dot(R, x_prev.T)
                #Z = Z + self.get_eta(t) * np.dot(R.reshape(self.neurons,1) , x_prev.reshape(1,self.neurons))
                R_sum += np.abs(R)

            e = (datetime.now() - start).seconds
            r = np.sum(R_sum) / (self.timepoints * self.batch_size)
            msg = 'T=%.4d E=%ds, R=%f' % (k, e, r)
            print msg
            f.write(msg +'\n')

            if k > 0 and k % self.log_often == 0:
                self.log_and_save('Z', Z, k == self.log_often)
                print 'Saved Z%d' % self.log_idx

        f.close()
        self.log_and_save('Z', Z)

    def old_train_dynamics(self):

        # Obj a - Za ** OLD **
        for t in range(self.num_trials):
            msg = 'T=%.4d, ' % t
            k = 0
            for log in self.logs:
                R_sum = np.zeros(log.shape[0])
                for i in range(1, log.shape[1]):
                    x_prev, x = (log[:, i-1], log[:, i])
                    R = x - np.dot(Z, x_prev)
                    Z = Z + self.get_eta(t) * np.dot(R.reshape(self.neurons,1) , x_prev.reshape(1,self.neurons))
                    R_sum += np.abs(R)
                msg += 'R%d = %.3f ' % (k, np.sqrt(np.sum(R_sum)))
                #msg += 'R%d = %.3f ' % (k, np.sqrt(np.sum(R_sum) / log.shape[1])) # Later
                k += 1

            print msg
            f.write(msg +'\n')
            log_often = 200
            if t > 0 and t % log_often  == 0:
                self.log_and_save(Z, t == log_often)
                print 'Saved Z%d' % self.log_idx

        f.close()
        self.log_and_save(Z)

    def temporal_correlation(self, coeffs, time, patch_i):
        'Make a scatter plot of the activity of a pair of coefficients and color by time point'
        if len(coeffs) != 2:
            raise Exception("Greater than 2 dim not supported.")

        tstart, tend = time[0], time[1]
        pi = patch_i
        data = self.log[coeffs, pi, tstart:tend]

        cartesian = False
        if True:
            fg, ax = plt.subplots(3,1)
            fg.set_size_inches(8.0,8.0)
            h = np.max(np.abs(data))
            ax[0].plot([-h, h], [0, 0], color='k')
            ax[0].plot([0, 0], [-h, h], color='k')

            colors = np.linspace(0, 1, tend - tstart)
            cmap = plt.get_cmap('autumn')

            #scat = plt.scatter(data[0], data[1], s=12, c=colors, cmap=cmap, lw=0)
            scat = ax[0].scatter(data[0], data[1], s=50, c=colors, lw=1)
            cbar = plt.colorbar(scat, ax=ax[0], ticks=[0, 0.5, 1])
            cbar.ax.set_yticklabels(['t=%d' % tstart, 't=%d' % np.average((tstart, tend)), 't=%d' % tend])
            ax[0].set_xlabel("A%d" % coeffs[0], fontdict={'fontsize':18})
            ax[0].set_ylabel("A%d" % coeffs[1], fontdict={'fontsize':18})
            ax[0].axis('equal')

            theta = np.arctan(data[0]/data[1])
            for i in range(len(theta)):
                if math.isnan(theta[i]):
                    theta[i] = 0
            #theta[math.isnan(theta)] = 0
            r     = np.sqrt(data[0] ** 2 + data[1] ** 2)

            ax[1].set_title('Radius')
            ax[1].plot(range(tend - tstart), r)
            ax[2].set_title('Theta')
            ax[2].plot(range(tend - tstart), theta)
            plt.show()

def get_neighbors(coeff, dist=2):
    # Retrieve the neighbor indices of a topographic group
    group = []

    for col in range(-dist, dist):
        for row in range(-dist, dist):
            group.append(coeff + col * 18 + row * 1)
    return group

#a.over_time()
#a.train_dynamics()
#a.train_G_dynamics()

def foo(patch, tstart, tend):
    return 1

def test(q, patch, tstart, tend):
    try:
        q.put((1,2))
    except Exception as e:
        traceback.print_exc()
        print 'e', e.value
        q.put((2,3))
    return -1

def over_time_2(image, phi_name, patch_per_dim, sz, phi, log, coeffs, patch, tstart, tend, q=None, time_only=False):
    'Plot the activity over time, the video, and the reconstruction, simultaneously'
    try:
        # Plot Setup
        tlen = tend - tstart
        row_start = 2
        rows = 1 if time_only else 6
        cols = 6

        fg = matplotlib.pyplot.figure(figsize=(10.0, 10.0))

        # Time graph
        ax_time = plt.subplot2grid((rows,cols), (0,0), rowspan=row_start, colspan=cols)
        height = 1.2 * np.max(np.abs(log[:, patch, tstart:tend]))
        ax_time.plot(range(tlen), [0] * tlen, color='k') # X axis
        ax_time.plot([0, 0], [-height, height], color='k') # Y axis
        for i in coeffs:
            ax_time.plot(range(tlen), log[i, patch, tstart:tend], label='A%d' % i)
        #ax_time.legend()
        #lg = ax_time.legend(bbox_to_anchor=(-0.6 , 0.40), loc=2, fontsize=10)
        #lg = ax_time.legend(bbox_to_anchor=(1.05, 1), loc=2, fontsize=50)
        lg = ax_time.legend(bbox_to_anchor=(0., 0, 1., .102), loc=3,
                   ncol=2, fontsize=10, borderaxespad=0.)
        lg.draw_frame(False)
        ax_time.axis([0, tlen - 1, -height, height])
        ax_time.set_title("Patch=%d, t=(%d,%d) Dict=%s" % (patch, tstart, tend, phi_name))

        if not time_only:
            # Coefficients
            i = 0
            for r in range(row_start+1, rows):
                for c in range(cols):
                    if i >= len(coeffs):
                        break
                    ax = plt.subplot2grid((rows,cols), (r,c))
                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)
                    ax.imshow(np.reshape(phi[:,coeffs[i]], (sz, sz)), cmap = cm.binary, interpolation='nearest')
                    ax.set_title("A%d" % coeffs[i])
                    i = i + 1

            # Reconstruction and Images
            direc = 'activity/images/'
            name = '%s_P%d_T%d.%d' % (phi_name, patch, tstart, tend)
            for iters in range(1): # Show forever
                for t in range(tstart, tend):
                    #for log in self.logs:
                    k = 0
                    col = int(np.floor(cols/2) - 1)
                    ax_r = plt.subplot2grid((rows,cols), (row_start, col)) # Reconstruct
                    img = np.dot(phi[:, coeffs], log[coeffs,patch,t])

                    ax_r.set_title("Rec t=%d" % t)
                    ax_r.get_xaxis().set_visible(False)
                    ax_r.get_yaxis().set_visible(False)

                    ax_r.imshow(np.reshape(img, (sz, sz)), cmap = cm.binary, interpolation='nearest')

                    col = int(np.floor((cols/2) + 1) - 1)
                    ax_i = plt.subplot2grid((rows,cols), (row_start, col)) # Images
                    ax_i.get_xaxis().set_visible(False)
                    ax_i.get_yaxis().set_visible(False)
                    ax_i.set_title("Img  t=%d" % t)

                    rr = (np.floor(patch / patch_per_dim)) * sz
                    cc = (patch % patch_per_dim) * sz

                    dimg = image[rr:rr+sz, cc:cc+sz, t].T
                    ax_i.imshow(dimg, cmap = cm.binary, interpolation='nearest')
                    plt.draw()
                    #plt.savefig('%sstage/%s_t=%3d.png' % (direc, name, t))
                    plt.savefig('%s/%s_t=%3d.png' % (direc, name, t))

        #os.system('convert -delay 40 -loop 0 %sstage/*.png activity/gifs/%s.gif' % (direc, name))
        os.system('convert -delay 40 -loop 0 %s/%s*.png activity/gifs/%s.gif' % (direc, name, name))
        #os.system('mv %sstage/*.png %s' % (direc, direc))
        plt.close()
        if q is not None:
            q.put(1)
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        if q is not None:
            q.put(e, exc_type, exc_obj, exc_tb)
        else:
            print e, exc_type, exc_obj, 'lineno', exc_tb.tb_lineno
            raise e
    #plt.show(block=True)



a = Analysis()
#a.survey_activity()
#a.temporal_correlation([219, 181], [0,20], 189)
a.temporal_correlation([219, 181], [0,7], 189)
