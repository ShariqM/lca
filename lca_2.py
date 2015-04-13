" Run LCA on a data set"
import matplotlib
import socket
if socket.gethostname() == 'redwood2':
    matplotlib.use('Agg') # Don't crash because $Display is not set correctly on the cluster

import pdb

from math import log
from math import isnan
import random
import scipy.io
import numpy as np
from datetime import datetime
import time

import matplotlib.pyplot as plt
from matplotlib import cm

import sys
import os
import h5py
import inspect
from theano import *
import theano.tensor as T

from showbfs import showbfs
from helpers import *
from runtype import *
from runp import *
from colors import COLORS

class LcaNetwork():

    datasets = {0:'IMAGES_FIELD',
                1:'IMAGES_DUCK_SHORT',
                2:'IMAGES_DUCK',
                3:'IMAGE_DUCK_LONG',
                4:'IMAGES_DUCK_120'}

    # Sparse Coding Parameters
    patch_dim    = 144 # patch_dim=(sz)^2 where the basis and patches are SZxSZ
    neurons      = 1024 # Number of basis functions
    sz           = np.sqrt(patch_dim)

    lambdav      = 0.15  # Minimum Threshold
    batch_size   = 100
    border       = 4
    num_trials   = 20000

    #init_phi_name = '' # Blank if you want to start from scratch
    init_phi_name = 'Phi_193_37.0.mat'

    # LCA Parameters
    skip_frames  = 80 # When running vLearning don't use the gradient for the first 80 iterations of LCA
    fixed_lambda = True # Don't initialize the threshold above lambdav and decay down
    lambda_decay = 0.95
    thresh_type  = 'soft'
    coeff_eta    = 0.07
    lambda_type  = ''
    group_sparse = 1     # Group Sparse Coding (1 is normal sparse coding)
    iters_per_frame = 30 # Only for vLearning
    time_batch_size = 100
    load_sequentially = False # Unsupported ATM. Don't grab random space-time boxes
    save_activity = False # Only supported for vReconstruct

    # General Parameters
    runtype            = RunType.vReconstruct # Learning, vLearning, vReconstruct

    # Visualizer parameters
    coeff_visualizer = True # Visualize potentials of neurons on a single patch
    iter_idx        = 0
    num_frames      = 2 # Number of frames to visualize
    if coeff_visualizer:
        matplotlib.rcParams.update({'figure.autolayout': True}) # Magical tight layout
    graphics_initialized = False
    save_cgraphs = False

    random_patch_index = 1                    # Patch for coeff_visualizer
    start_t = 0                               # Used if you want to continue learning of an existing dictionary

    def __init__(self):
        self.image_data_name = self.datasets[1]
        self.IMAGES = self.get_images(self.image_data_name)
        (self.imsize, imsize, self.num_images) = np.shape(self.IMAGES)
        self.patch_per_dim = int(np.floor(imsize / self.sz))

        if self.coeff_visualizer:
            self.batch_size = 1
        print 'num images %d, num trials %d' % (self.num_images, self.num_trials)

        # Don't block when showing images
        plt.ion()

    def get_images(self, image_data_name):
        if 'LONG' in image_data_name or '120' in image_data_name:
            f = h5py.File('mat/%s.mat' % image_data_name, 'r',) # Need h5py for big file
            IMAGES = np.array(f.get(image_data_name))
            IMAGES = np.swapaxes(IMAGES, 0, 2) # v7.3 reorders for some reason, or h5?
        else:
            IMAGES = scipy.io.loadmat('mat/%s.mat' % image_data_name)[image_data_name]
        return IMAGES

    # Image Loaders
    def load_rimages(I):
        '(1) Choose a batch_size number of random images. Used by Learning()'

        imi = np.ceil(num_images * random.uniform(0, 1))
        # Pick batch_size random patches from the random image
        for i in range(batch_size):
            r = border + np.ceil((imsize-sz-2*border) * random.uniform(0, 1))
            c = border + np.ceil((imsize-sz-2*border) * random.uniform(0, 1))

            I[:,i] = np.reshape(IMAGES[r:r+sz, c:c+sz, imi-1], patch_dim, 1)

        return I

    def load_image(self, I, t):
        '(2) Load all patches for image $t$. Used by vReconstruct()'

        sz = self.sz
        if self.coeff_visualizer: # Pick 1 patch insteadj
            rr = self.random_patch_index * sz
            cc = self.random_patch_index * sz
            I[:,0] = np.reshape(self.IMAGES[rr:rr+sz, cc:cc+sz, t], self.patch_dim, 1)
        else:
            i = 0
            for r in range(self.patch_per_dim):
                for c in range(self.patch_per_dim):
                    rr = r * sz
                    cc = c * sz
                    I[:,i] = np.reshape(self.IMAGES[rr:rr+sz, cc:cc+sz, t], self.patch_dim, 1)
                    i = i + 1
        return I

    def load_videos():
        '(3) Load a batch_size number of Space-Time boxes of individual random patches. Used by vLearning()'
        VI = np.zeros((patch_dim, batch_size, time_batch_size))
        #TODO do I need a coeff_visualizer here?
        for x in range(batch_size):
            # Choose a random image less than time_batch_size images away from the end
            imi = np.floor((num_images - time_batch_size) * random.uniform(0, 1))
            r = border + np.ceil((imsize-sz-2*border) * random.uniform(0, 1))
            c = border + np.ceil((imsize-sz-2*border) * random.uniform(0, 1))
            VI[:,x,:] = np.reshape(IMAGES[r:r+sz, c:c+sz, imi:imi+time_batch_size], (patch_dim, time_batch_size), 1)
        return VI

    def init_Phi(self):
        # Initialize basis functions
        if self.init_phi_name != '':
            Phi = scipy.io.loadmat('dict/%s' % self.init_phi_name)
            Phi = Phi['Phi']
        else:
            Phi = np.random.randn(patch_dim, neurons)
            Phi = np.dot(Phi, np.diag(1/np.sqrt(np.sum(Phi**2, axis = 0))))
        return Phi


    def Learning(self):
        'Run the normal, no inertia, LCA learning algorithm'
        Phi = self.init_Phi()

        # Initialize batch of images
        I = np.zeros((patch_dim, batch_size))

        max_active = float(neurons * batch_size)
        start = datetime.now()

        if runtype == RunType.Learning:
            for t in range(start_t, num_trials):
                I = load_images(I)
                u, ahat = self.sparsify(I) # Coefficient Inference

                # Calculate Residual
                R = I - tdot(Phi, ahat)

                # Update Basis Functions
                dPhi = get_eta(t, neurons, runtype, batch_size) * (tdot(R, ahat.T))

                Phi = Phi + dPhi
                Phi = tdot(Phi, np.diag(1/np.sqrt(np.sum(Phi**2, axis = 0))))

                # Plot every 200 iterations
                if np.mod(t, 20  ) == 0:
                    var = I.var().mean()
                    mse = (R ** 2).mean()
                    snr = 10 * log(var/mse, 10)

                    ahat_c = np.copy(ahat)
                    ahat_c[np.abs(ahat_c) > lambdav/1000.0] = 1
                    ac = np.sum(ahat_c)

                    print '%.4d) lambdav=%.3f || snr=%.2fdB || AC=%.2f%% || ELAP=%d' \
                            % (t, lambdav, snr, 100.0 * ac / max_active,
                               (datetime.now() - start).seconds)

                    sys.stdout.flush()
                    showbfs(Phi, get_eta(t, neurons, runtype, batch_size))
                    plt.show()

                if np.mod(t, 20) == 0:
                    log_and_save_dict(Phi, 100.0 * float(t)/num_trials)

        log_and_save_dict(Phi, 100.0)
        plt.show()

    def vLearning(self):
        'Run the video, inertia, LCA learning algorithm'
        Phi = self.init_Phi()

        # Initialize batch of images
        I = np.zeros((patch_dim, batch_size))

        max_active = float(neurons * batch_size)
        start = datetime.now()

        for t in range(start_t, num_trials):
            VI = load_videos()

            u_prev = None # Neurons keep state from previous frame
            for i in range(time_batch_size):
                I = VI[:,:,i]

                u, ahat = sparsify(I, Phi, lambdav, u_prev=u_prev,
                                    num_iterations=iters_per_frame)
                u_prev = u

                # Calculate Residual
                R = I - tdot(Phi, ahat)

                if i >= skip_frames/iters_per_frame: # Don't learn on the first skip_frames
                    # Update Basis Functions
                    dPhi = get_veta(batch_size * t, neurons, runtype, time_batch_size) * (tdot(R, ahat.T))

                    Phi = Phi + dPhi
                    Phi = tdot(Phi, np.diag(1/np.sqrt(np.sum(Phi**2, axis = 0))))

                ahat_c = np.copy(ahat)
                ahat_c[np.abs(ahat_c) > lambdav/1000.0] = 1
                ac = np.sum(ahat_c)

                if i % 50 == 0:
                    print '\t%.3d) lambdav=%.3f || AC=%.2f%%' % (i, lambdav, 100.0 * ac / max_active)

            var = I.var().mean()
            mse = (R ** 2).mean()
            snr = 10 * log(var/mse, 10)

            sys.stdout.flush()

            if np.mod(t, 5) == 0:
                showbfs(Phi, get_veta(batch_size * t, neurons, runtype, time_batch_size))
                log_and_save_dict(Phi, 100.0 * float(t)/num_trials)
            plt.show()

            print '%.4d) lambdav=%.3f || snr=%.2fdB || AC=%.2f%% || ELAP=%d' \
                        % (t, lambdav, snr, 100.0 * ac / max_active,
                           (datetime.now() - start).seconds)

        log_and_save_dict(Phi, 100.0)
        plt.show()

    def vReconstruct(self):
        # Load dict from Learning run
        Phi = scipy.io.loadmat('dict/%s' % self.init_phi_name)['Phi']

        # Set the batch_size to number of patches in an image
        if not self.coeff_visualizer:
            self.batch_size = self.patch_per_dim**2

        # Initialize batch of images
        I = np.zeros((self.patch_dim, self.batch_size))

        max_active = float(self.neurons * self.batch_size)

        run_p = [RunP(True, 10, self.lambdav)]
        labels = get_labels(run_p)

        runs = len(labels)
        rcolor = [COLORS['red'], COLORS['green'], COLORS['blue'], COLORS['black'],
                  COLORS['yellow2'], COLORS['purple']]
        top_CC_AC = 0 # [C]hanged [C]oefficients, [A]ctive [C]oefficients

        for run in range(runs):
            # Record data
            MSE = [] # Mean Squared Error
            SNR = [] # Signal to Noise ratio
            AC  = np.zeros(self.num_frames) # Active coefficients
            CC  = np.zeros(self.num_frames) # Changing coefficients
            activity_log = np.zeros((self.neurons, self.batch_size, self.num_frames))

            u_prev = None
            ahat_prev = np.zeros((self.neurons, self.batch_size))
            ahat_prev_c = np.zeros((self.neurons, self.batch_size))

            start = datetime.now()
            for t in range(self.num_frames):
                I = self.load_image(I, t)

                if run_p[run].initP == True:
                    u, ahat = self.sparsify(I, Phi, u_prev=u_prev, num_iterations=run_p[run].iters)
                else:
                    u, ahat = self.sparsify(I, Phi, num_iterations=run_p[run].iters)

                activity_log[:,:,t] = ahat

                # Calculate Residual Error
                R = I - tdot(Phi, ahat)
                mse = (R ** 2).mean()
                MSE.append(mse)
                var = I.var().mean()
                SNR.append(10 * log(var/mse, 10))

                ahat_prev = ahat

                ahat_c = np.copy(ahat)
                ahat_c[np.abs(ahat_c) > self.lambdav/1000.0] = 1
                AC[t] = np.sum(ahat_c)
                CC[t] = np.sum(ahat_c!=ahat_prev_c)
                ahat_prev_c = ahat_c
                u_prev = u

                print '%.3d) %s || lambdav=%.3f || snr=%.2fdB || AC=%.2f%%' \
                        % (t, labels[run], self.lambdav, SNR[t], 100.0 * AC[t] / max_active)
            elapsed = (datetime.now() - start).seconds

            if self.save_activity:
                np.save('coeff_S193__initP20_100t', activity_log)
            plt.close()
            matplotlib.rcParams.update({'figure.autolayout': False})

            plt.subplot(231)
            plt.xlabel('Time (steps)', fontdict={'fontsize':12})
            plt.ylabel('MSE', fontdict={'fontsize':12})
            plt.axis([0, self.num_frames, 0.0, max(MSE) * 1.1])
            plt.plot(range(self.num_frames), MSE, color=rcolor[run], label=labels[run])
            lg = plt.legend(bbox_to_anchor=(-0.6 , 0.40), loc=2, fontsize=10)
            lg.draw_frame(False)

            top_CC_AC = max(max(CC),max(AC),top_CC_AC) * 1.1
            plt.subplot(232)
            plt.xlabel('Time (steps)', fontdict={'fontsize':12})
            plt.ylabel('# Active Coeff', fontdict={'fontsize':12})
            plt.axis([0, self.num_frames, 0, top_CC_AC])
            plt.plot(range(self.num_frames), AC, color=rcolor[run])

            plt.subplot(233)
            plt.xlabel('Time (steps)', fontdict={'fontsize':12})
            plt.ylabel('# Changed Coeff', fontdict={'fontsize':12})
            plt.axis([0, self.num_frames, 0, top_CC_AC])
            plt.plot(range(self.num_frames), CC, color=rcolor[run])

            plt.subplot(234)
            plt.xlabel('Time (steps)', fontdict={'fontsize':12})
            plt.ylabel('SNR (dB)', fontdict={'fontsize':12})
            plt.axis([0, self.num_frames, 0.0, 22])
            plt.plot(range(self.num_frames), SNR, color=rcolor[run], label=labels[run])
            lg = plt.legend(bbox_to_anchor=(-0.6 , 0.60), loc=2, fontsize=10)
            lg.draw_frame(False)

            # % plots
            top_p = 100 * top_CC_AC / max_active
            plt.subplot(235)
            plt.xlabel('Time (steps)', fontdict={'fontsize':12})
            plt.ylabel('% Active Coeff', fontdict={'fontsize':12})
            plt.axis([0, self.num_frames, 0, top_p])
            plt.plot(range(self.num_frames), 100 * AC / max_active, color=rcolor[run])

            plt.subplot(236)
            plt.xlabel('Time (steps)', fontdict={'fontsize':12})
            plt.ylabel('% Changed Coeff', fontdict={'fontsize':12})
            plt.axis([0, self.num_frames, 0, top_p])
            plt.plot(range(self.num_frames), 100 * CC / max_active, color=rcolor[run])

        plt.suptitle("DATA=%s, LAMBDAV=%.3f, IMG=%dx%d, PAT=%dx%d, DICT=%d, PAT/IMG=%d ELAP=%d" %
                        (self.image_data_name, self.lambdav, self.imsize, self.imsize,
                        self.sz, self.sz, self.neurons, self.batch_size, elapsed), fontsize=18)
        plt.show(block=True)


    def sparsify(self, I, Phi, u_prev=None, num_iterations=80):
        'Run the LCA coefficient dynamics'

        b = tdot(Phi.T, I)
        G = tdot(Phi.T, Phi) - np.eye(self.neurons)

        if self.fixed_lambda:
            l = np.ones(self.batch_size)
            l *= self.lambdav
            self.lambda_type = 'Fixed and lambdav'
        else:
            l = 0.5 * np.max(np.abs(b), axis = 0)
            self.lambda_type = 'l = 0.5 * np.max(np.abs(b), axis = 0)'

        u = u_prev if u_prev is not None else np.zeros((self.neurons, self.batch_size))
        a = self.thresh(u, l)

        showme = True

        #if not initialized and coeff_visualizer:
        if self.coeff_visualizer:
            if not self.graphics_initialized:
                self.graphics_initialized = True
                fg, self.ax = plt.subplots(2,2)

                self.ax[1,1].set_title('Coefficients')
                self.ax[1,1].set_xlabel('Coefficient Index')
                self.ax[1,1].set_ylabel('Activity')

                self.coeffs = self.ax[1,1].bar(range(self.neurons), np.abs(u), color='r', lw=0)
                self.lthresh = self.ax[1,1].plot(range(self.neurons+1), list(l) * (self.neurons+1), color='g')

                axis_height = 1.05 if self.runtype == RunType.Learning else self.lambdav * 5
                self.ax[1,1].axis([0, self.neurons, 0, axis_height])

                recon = tdot(Phi, a)
                self.ax[0,0].imshow(np.reshape(recon, (self.sz, self.sz)),cmap = cm.binary, interpolation='nearest')
                self.ax[0,0].set_title('Iter=%d\nReconstruct' % 0)

                self.ax[0,1].set_title('Reconstruction Error')
                self.ax[0,1].set_xlabel('Time (steps)')
                self.ax[0,1].set_ylabel('SNR (dB)')
                self.ax[0,1].axis([0, self.num_frames * num_iterations, 0.0, 22])

                # The subplots move around if I don't do this lol...
                for i in range(6):
                    plt.savefig('animation/junk.png')
                if self.save_cgraphs:
                    plt.savefig('animation/%d.jpeg' % self.iter_idx)
                self.iter_idx += 1

            self.ax[1,0].imshow(np.reshape(I[:,0], (self.sz, self.sz)),cmap = cm.binary, interpolation='nearest')
            self.ax[1,0].set_title('Image')

            if showme:
                plt.draw()
                plt.show()

        for t in range(num_iterations):
            u = self.coeff_eta * (b - tdot(G,a)) + (1 - self.coeff_eta) * u
            a = self.thresh(u, l)

            check_activity(b, G, u, a)

            l = self.lambda_decay * l
            l[l < self.lambdav] = self.lambdav

            if self.coeff_visualizer:
                # Update the potentials
                for coeff, i in zip(self.coeffs, range(self.neurons)):
                    coeff.set_height(abs(u[i]))
                self.lthresh[0].set_data(range(self.neurons+1), list(l) * (self.neurons+1))

                # Update Reconstruction
                recon = tdot(Phi, a)
                self.ax[0,0].imshow(np.reshape(recon, (self.sz, self.sz)),cmap = cm.binary, interpolation='nearest')
                self.ax[0,0].set_title('Iter=%d\nReconstruct' % self.iter_idx)

                # Plot SNR
                var = I.var().mean()
                R = I - recon
                mse = (R ** 2).mean()
                snr = 10 * log(var/mse, 10)
                color = 'r' if t == 0 else 'g'
                self.ax[0,1].scatter(self.iter_idx, snr, s=8, c=color)

                if showme:
                    plt.draw()
                    plt.show()
                if self.save_cgraphs:
                    plt.savefig('animation/%d.jpeg' % self.iter_idx)
                self.iter_idx += 1

        return u, a

    def group_thresh(self, a, theta):
        if len(a) % self.group_sparse != 0:
            raise Exception('len(a) %% group_sparse = %d' % (len(a) % selfgroup_sparse))
        size = len(a)/self.group_sparse
        chunks = [np.array(a[x:x+size]) for x in range(0, len(a), size)]
        for c in chunks:
            c[np.sum(np.abs(chunks), axis=0) < theta] = 0
        return np.concatenate(chunks)

    def thresh(self, u, theta):
        'LCA threshold function'
        if self.thresh_type=='hard': # L0 Approximation
            a = u;
            if group_sparse > 1:
                a = self.group_thresh(u, theta)
            else:
                a[np.abs(a) < theta] = 0
        elif self.thresh_type=='soft': # L1 Approximation
            a = abs(u) - theta;
            a[a < 0] = 0
            a = np.sign(u) * a
        return a

    def get_phi_idx(self):
        f = open('log.txt', 'r')
        rr = 0
        while True:
            r = f.readline()
            if r == '':
                break
            rr = r
        f.close()
        return int(rr)

    def log_and_save_dict(self, Phi, comp):
        'Log dictionary and Save Mat file'
        phi_idx = self.get_phi_idx()

        if comp == 0.0: # Only log on first write
            name = 'Phi_%d' % (phi_idx + 1)

            path = 'dict/%s' % name
            if not os.path.exists(path):
                os.makedirs(path)

            f = open('log.txt', 'a') # append to the log

            f.write('\n*** %s ***\n' % name)
            f.write('Time=%s\n' % datetime.now())
            f.write('RunType=%s\n' % get_RunType_name(runtype))
            f.write('IMAGES=%s\n' % image_data_name)
            f.write('patch_dim=%d\n' % patch_dim)
            f.write('neurons=%d\n' % neurons)
            f.write('lambdav=%.3f\n' % lambdav)
            f.write('Lambda Decay=%.2f\n' % lambda_decay)
            f.write('num_trials=%d\n' % num_trials)
            f.write('batch_size=%d\n' % batch_size)
            f.write('time_batch_size=%d\n' % time_batch_size)
            f.write('NUM_IMAGES=%d\n' % num_images)
            f.write('iter_per_frame=%d\n' % iters_per_frame)
            f.write('thresh_type=%s\n' % thresh_type)
            f.write('coeff_eta=%.3f\n' % self.coeff_eta)
            f.write('lambda_type=[%s]\n' % lambda_type)
            f.write('InitPhi=%s\n' % init_phi_name)
            f.write('Load Sequentially=%s\n' % load_sequentially)
            f.write('Skip initial frames=%s\n' % skip_frames)
            f.write('Group Sparse=%d\n' % group_sparse)

            f.write('%d\n' % (int(rr)+1))
            f.close()

            logfile = '%s/%s_log.txt' % (path, name)
            print 'Assigning stdout to %s' % logfile
            sys.stdout = open(logfile, 'w') # Rewire stdout to write the log

            # print these methods so we know the simulated annealing parameters
            print inspect.getsource(get_eta)
            print inspect.getsource(get_veta)
        else:
            name = 'Phi_%d' % (int(rr))
            path = 'dict/%s' % name

        fname = '%s/%s_%.1f' % (path, name, comp)
        plt.savefig('%s.png' % fname)
        scipy.io.savemat(fname, {'Phi':Phi})
        print '%s_%.1f successfully written.' % (name, comp)

    def run(self):
        if self.runtype == RunType.vReconstruct:
            self.vReconstruct()
        elif self.runtype == Runtype.Learning:
            self.Learning()
        else:
            self.vLearning()

lca = LcaNetwork()
lca.run()
