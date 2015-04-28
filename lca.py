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
                4:'IMAGES_DUCK_120',
                5:'IMAGES_MOVE_RIGHT',
                6:'IMAGES_BOUNCE',
                7:'IMAGES_PATCH_DUCK',
                8:'IMAGES_EDGE_DUCK',
                9:'IMAGES_EDGE_RIGHT_DUCK'}

    # Sparse Coding Parameters
    patch_dim    = 144 # patch_dim=(sz)^2 where the basis and patches are SZxSZ
    #neurons      = patch_dim * 1 # Number of basis functions
    neurons      = 144 # Number of basis functions
    #neurons      = patch_dim * 8 # Number of basis functions
    sz           = np.sqrt(patch_dim)

    # Typical lambda is 0.07 for reconstruct, 0.15 for learning
    lambdav      = 0.15   # Minimum Threshold
    batch_size   = 100
    border       = 4
    num_trials   = 20000

    init_phi_name = '' # Blank if you want to start from scratch
    #init_phi_name = 'Phi_271_2.7.mat'
    #init_phi_name = 'Phi_272_0.5.mat'
    #init_phi_name = 'Phi_289/Phi_289_0.4.mat'

    # LCA Parameters
    skip_frames  = 80 # When running vLearning don't use the gradient for the first 80 iterations of LCA
    fixed_lambda = True # Don't initialize the threshold above lambdav and decay down
    lambda_decay = 0.95
    thresh_type  = 'soft'
    #coeff_eta    = 0.25 # Wack point
    coeff_eta    = 0.05 # Normal
    #coeff_eta    = 0.01 # Normal
    u_factor = 0.8

    lambda_type  = ''
    group_sparse = 1     # Group Sparse Coding (1 is normal sparse coding)
    iters_per_frame = 10  # Only for vLearning
    time_batch_size = 100
    load_sequentially = False # Unsupported ATM. Don't grab random space-time boxes
    save_activity = False # Only supported for vReconstruct

    # General Parameters
    runtype            = RunType.Learning # Learning, vLearning, vmLearning, vReconstruct
    #runtype            = RunType.vmLearning # Learning, vLearning, vmLearning, vReconstruct
    log_and_save = False # Log parameters save dictionaries

    # Visualizer parameters
    coeff_visualizer = False # Visualize potentials of neurons on a single patch
    iter_idx        = 0
    num_frames      = 100 # Number of frames to visualize
    #num_coeff_upd   = num_frames * iters_per_frame # This is correct when vPredict is off
    lb4predict      = 8 # Number of frames to let the dynamics settle before predicting
    num_coeff_upd   = lb4predict * 10 + num_frames - lb4predict #  Special case for vPredict

    if coeff_visualizer:
        matplotlib.rcParams.update({'figure.autolayout': True}) # Magical tight layout
    graphics_initialized = False
    save_cgraphs = False

    random_patch_index = 3                    # Patch for coeff_visualizer
    start_t = 0                               # Used if you want to continue learning of an existing dictionary

    def __init__(self):
        self.image_data_name = self.datasets[9]
        self.IMAGES = self.get_images(self.image_data_name)
        (self.imsize, imsize, self.num_images) = np.shape(self.IMAGES)
        self.patch_per_dim = int(np.floor(imsize / self.sz))

        try:
            self.phi_idx = self.get_phi_idx()
        except Exception as e:
            raise Exception('Corrupted log.txt file, please fix manually')

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
    def load_rimages(self, I):
        '(1) Choose a batch_size number of random images. Used by Learning()'

        border, imsize, sz = self.border, self.imsize, self.sz
        imi = np.ceil(self.num_images * random.uniform(0, 1))
        # Pick batch_size random patches from the random image
        for i in range(self.batch_size):
            r = border + np.ceil((imsize-sz-2*border) * random.uniform(0, 1))
            c = border + np.ceil((imsize-sz-2*border) * random.uniform(0, 1))

            I[:,i] = np.reshape(self.IMAGES[r:r+sz, c:c+sz, imi-1], self.patch_dim, 1)

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

    def load_videos(self):
        '(3) Load a batch_size number of Space-Time boxes of individual random patches. Used by vLearning()'

        border, imsize, sz, tbs = self.border, self.imsize, self.sz, self.time_batch_size
        VI = np.zeros((self.patch_dim, self.batch_size, self.time_batch_size))
        if self.coeff_visualizer:
            r = self.random_patch_index * sz
            c = self.random_patch_index * sz
            VI[:,0,:] = np.reshape(self.IMAGES[r:r+sz, c:c+sz, 0:tbs], (self.patch_dim, tbs), 1)
        else:
            for x in range(self.batch_size):
                # Choose a random image less than time_batch_size images away from the end
                imi = np.floor((self.num_images - self.time_batch_size) * random.uniform(0, 1))
                r = border + np.ceil((imsize-sz-2*border) * random.uniform(0, 1))
                c = border + np.ceil((imsize-sz-2*border) * random.uniform(0, 1))
                VI[:,x,:] = np.reshape(self.IMAGES[r:r+sz, c:c+sz, imi:imi+tbs], (self.patch_dim, tbs), 1)
        return VI

    def init_Phi(self):
        # Initialize basis functions
        if self.init_phi_name != '':
            Phi = scipy.io.loadmat('dict/%s' % self.init_phi_name)
            Phi = Phi['Phi']
        else:
            Phi = np.random.randn(self.patch_dim, self.neurons)
            Phi = np.dot(Phi, np.diag(1/np.sqrt(np.sum(Phi**2, axis = 0))))
        return Phi

    def Learning(self):
        'Run the normal, no inertia, LCA learning algorithm'
        Phi = self.init_Phi()

        # Initialize batch of images
        I = np.zeros((self.patch_dim, self.batch_size))

        max_active = float(self.neurons * self.batch_size)
        start = datetime.now()

        for t in range(self.start_t, self.num_trials):
            I = self.load_rimages(I)
            u, ahat = self.sparsify(I, Phi) # Coefficient Inference

            # Calculate Residual
            R = I - t2dot(Phi, ahat)

            # Update Basis Functions
            dPhi = get_eta(t, self.neurons, self.runtype, self.batch_size) * (t2dot(R, ahat.T))

            Phi = Phi + dPhi
            Phi = t2dot(Phi, np.diag(1/np.sqrt(np.sum(Phi**2, axis = 0))))

            # Plot every 200 iterations
            if np.mod(t, 20  ) == 0:
                var = I.var().mean()
                mse = (R ** 2).mean()
                snr = 10 * log(var/mse, 10)

                ahat_c = np.copy(ahat)
                ahat_c[np.abs(ahat_c) > self.lambdav/1000.0] = 1
                ac = np.sum(ahat_c)

                print '%.4d) lambdav=%.3f || snr=%.2fdB || AC=%.2f%% || ELAP=%d' \
                        % (t, self.lambdav, snr, 100.0 * ac / max_active,
                           (datetime.now() - start).seconds)

                sys.stdout.flush()
                showbfs(Phi, self.phi_idx)
                plt.show()

            if np.mod(t, 20) == 0:
                self.view_log_save(Phi, 100.0 * float(t)/self.num_trials)

        self.view_log_save(Phi, 100.0)

    def vLearning(self):
        'Run the video, inertia, LCA learning algorithm'
        Phi = self.init_Phi()

        # Initialize batch of images
        I = np.zeros((self.patch_dim, self.batch_size))

        max_active = float(self.neurons * self.batch_size)
        start = datetime.now()

        for t in range(self.start_t, self.num_trials):
            VI = self.load_videos()

            u_pred = None # Neurons keep state from previous frame
            for i in range(self.time_batch_size):
                I = VI[:,:,i]

                u, ahat = self.sparsify(I, Phi, u_pred=u_pred, num_iterations=self.iters_per_frame)
                u_pred = u

                # Calculate Residual
                R = I - tdot(Phi, ahat)

                if i >= self.skip_frames/self.iters_per_frame: # Don't learn on the first skip_frames
                    # Update Basis Functions
                    dPhi = get_veta(self.batch_size * t, self.neurons,
                                    self.runtype, self.time_batch_size) * (tdot(R, ahat.T))

                    Phi = Phi + dPhi
                    Phi = tdot(Phi, np.diag(1/np.sqrt(np.sum(Phi**2, axis = 0))))

                ahat_c = np.copy(ahat)
                ahat_c[np.abs(ahat_c) > self.lambdav/1000.0] = 1
                ac = np.sum(ahat_c)

                if i % 50 == 0:
                    print '\t%.3d) lambdav=%.3f || AC=%.2f%%' % (i, self.lambdav, 100.0 * ac / max_active)

            var = I.var().mean()
            mse = (R ** 2).mean()
            snr = 10 * log(var/mse, 10)

            sys.stdout.flush()

            if np.mod(t, 5) == 0:
                self.view_log_save(Phi, 100.0 * float(t)/self.num_trials)

            print '%.4d) lambdav=%.3f || snr=%.2fdB || AC=%.2f%% || ELAP=%d' \
                        % (t, self.lambdav, snr, 100.0 * ac / max_active,
                           (datetime.now() - start).seconds)

        self.view_log_save(Phi, 100.0)

    def vmLearning(self):
        'Run the video, inertia, transformation, LCA learning algorithm'
        Phi = self.init_Phi()

        # Initialize batch of images
        I = np.zeros((self.patch_dim, self.batch_size))

        # Transformation matrix
        Z = np.eye(self.neurons)
        #Z = initZ(self.neurons)
        #Z = np.random.randn(self.neurons, self.neurons)
        #Z = np.random.normal(0, 0.25, (self.neurons, self.neurons))
        #Z = np.eye(self.neurons) + np.random.normal(0, 0.25, (self.neurons, self.neurons))

        max_active = float(self.neurons * self.batch_size)
        start = datetime.now()

        for t in range(self.start_t, self.num_trials):
            VI = self.load_videos()

            u_prev = np.zeros((self.neurons, self.batch_size))
            u_pred = np.zeros((self.neurons, self.batch_size))
            a_pred = np.zeros((self.neurons, self.batch_size))
            for i in range(0, self.time_batch_size - 1):
                I = VI[:,:,i]

                u, ahat = self.sparsify(I, Phi, u_pred=u_pred, num_iterations=self.iters_per_frame)
                #if False and t > 15 and i > 20:
                    #u, ahat = u_pred, self.thresh(u_pred, np.ones(self.batch_size) * self.lambdav)
                #else:
                    #u, ahat = self.sparsify(I, Phi, u_pred=u_pred, num_iterations=self.iters_per_frame)

                # Calculate Residual
                R = I - t2dot(Phi, ahat)

                # Prediction Residual
                UR = u - u_pred
                ZR = I - t2dot(Phi, a_pred)
                #ZR = nI - t3dot(Phi, Z, ahat)

                if i >= self.skip_frames/self.iters_per_frame: # Don't learn on the first skip_frames
                    # Calculate dPhi
                    eta = get_veta(self.batch_size * t, self.neurons,
                                   self.runtype, self.time_batch_size)
                    Zahat = t2dot(Z, ahat)
                    dPhi =  eta * (t2dot(R, ahat.T))

                    # Calculate dZ
                    eta /= 8   # Hmmmmm...
                    dZ = eta * t2dot(UR, u_prev.T)

                    # Update
                    Phi = Phi + dPhi # Don't change Phi before calculating dZ!!!
                    Phi = t2dot(Phi, np.diag(1/np.sqrt(np.sum(Phi**2, axis=0))))
                    Z = Z + dZ

                    # Check new error
                    R2 = I - t2dot(Phi, ahat)
                    tmp_u_pred = t2dot(Z, u_prev) # Recalculate
                    tmp_a_pred = self.thresh(u_pred, np.ones(self.batch_size) * self.lambdav)
                    UR2 = u - tmp_u_pred
                    nUR = np.linalg.norm(UR)
                    nUR2 = np.linalg.norm(UR2)
                    if nUR2 > nUR:
                        print 'Fail: UR: %f, UR2: %f' % (np.linalg.norm(UR), np.linalg.norm(UR2))
                    ZR2 = I - t2dot(Phi, tmp_a_pred)
                    IR = I - t2dot(Phi, self.thresh(u_prev, np.ones(self.batch_size) * self.lambdav))

                ahat_c = np.copy(ahat)
                ahat_c[np.abs(ahat_c) > self.lambdav/1000.0] = 1
                ac = np.sum(ahat_c)

                if i % 50 == 0:
                    print '\t%.3d) lambdav=%.3f || AC=%.2f%%' % (i, self.lambdav, 100.0 * ac / max_active)
                u_prev = u
                u_pred = t2dot(Z, u_prev)
                a_pred = self.thresh(u_pred, np.ones(self.batch_size) * self.lambdav)


            var = I.var().mean()
            mse = (R ** 2).mean()
            snr = 10 * log(var/mse, 10)

            mse_a = (R2 ** 2).mean()
            snr_a = 10 * log(var/mse_a, 10)

            p_mse = (ZR ** 2).mean()
            p_snr = 10 * log(var/p_mse, 10)

            p_mse_a = (ZR2 ** 2).mean()
            p_snr_a = 10 * log(var/p_mse_a, 10)

            i_mse = (IR ** 2).mean()
            i_snr = 10 * log(var/i_mse, 10)

            sys.stdout.flush()

            if np.mod(t, 5) == 0:
                self.view_log_save(Phi, 100.0 * float(t)/self.num_trials, Z)

            print '%.4d) lambdav=%.3f || snr=%.2fdB || snr_a=%.2fdB || i_snr=%.2fdB || p_snr=%.2fdB || p_snr_a=%.2fdB || AC=%.2f%% || ELAP=%d' \
                        % (t, self.lambdav, snr, snr_a, i_snr, p_snr, p_snr_a, 100.0 * ac / max_active,
                           (datetime.now() - start).seconds)

        self.view_log_save(Phi, 100.0, Z)

    def vPredict(self):
        # Load dict from Learning run
        Phi = scipy.io.loadmat('dict/%s' % self.init_phi_name)['Phi']
        Z   = scipy.io.loadmat('dict/%s' % self.init_phi_name)['Z']

        # Set the batch_size to number of patches in an image
        if not self.coeff_visualizer:
            self.batch_size = self.patch_per_dim**2

        # Initialize batch of images
        I = np.zeros((self.patch_dim, self.batch_size))

        u_pred = None
        for t in range(self.num_frames):
            print 'herro'
            I = self.load_image(I, t)

            predict_mode = False
            if t < self.lb4predict:
                # Let the system settle to the correct dynamics
                u, ahat = self.sparsify(I, Phi, u_pred=u_pred, num_iterations=10)
            else:
                predict_mode = True
                print 'Predict'
                u, ahat = self.sparsify(I, Phi, u_pred=u_pred, num_iterations=0)
            #activity_log[:,:,t] = ahat

            # Calculate Residual Error
            R = I - t2dot(Phi, ahat)
            mse = (R ** 2).mean()
            var = I.var().mean()
            snr = 10 * log(var/mse, 10)

            name = 'Predicting' if predict_mode else 'Loading...'
            print '%.3d) %s || lambdav=%.3f || snr=%.2fdB'\
                    % (t, name, self.lambdav, snr)

            u_prev = u
            u_pred = t2dot(Z, u_prev)

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

            u_pred = None
            ahat_prev = np.zeros((self.neurons, self.batch_size))
            ahat_prev_c = np.zeros((self.neurons, self.batch_size))

            start = datetime.now()
            for t in range(self.num_frames):
                I = self.load_image(I, t)

                if run_p[run].initP == True:
                    u, ahat = self.sparsify(I, Phi, u_pred=u_pred, num_iterations=run_p[run].iters)
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
                u_pred = u

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

    def msparsify(self, I, nI, Phi, Z, u_pred=None, num_iterations=80):
        'Run the LCA coefficient dynamics'

        b = tdot(Phi.T, I)
        G = tdot(Phi.T, Phi) - np.eye(self.neurons)

        ZPhi = tdot(Phi, Z)
        Zb = tdot(ZPhi.T, nI)
        ZG = tdot(ZPhi.T, ZPhi) - np.eye(self.neurons)

        if self.fixed_lambda:
            l = np.ones(self.batch_size)
            l *= self.lambdav
            self.lambda_type = 'Fixed and lambdav'
        else:
            l = 0.5 * np.max(np.abs(b), axis = 0)
            self.lambda_type = 'l = 0.5 * np.max(np.abs(b), axis = 0)'

        u = u_pred if u_pred is not None else np.zeros((self.neurons, self.batch_size))
        a = self.thresh(u, l)

        showme = True # set to false if you just want to save the images
        if self.coeff_visualizer:
            if not self.graphics_initialized:
                self.graphics_initialized = True
                fg, self.ax = plt.subplots(3,3, figsize=(10,10))
                #fg.set_size_inches(08.0,8.0)


                self.ax[1,2].set_title('Coefficients')
                self.ax[1,2].set_xlabel('Coefficient Index')
                self.ax[1,2].set_ylabel('Activity')

                self.coeffs = self.ax[1,2].bar(range(self.neurons), np.abs(u), color='r', lw=0)
                self.lthresh = self.ax[1,2].plot(range(self.neurons+1), list(l) * (self.neurons+1), color='g')

                axis_height = 1.05 if self.runtype == RunType.Learning else self.lambdav * 5
                self.ax[1,2].axis([0, self.neurons, 0, axis_height])

                # Present
                recon = tdot(Phi, a)
                self.ax[1,1].imshow(np.reshape(recon, (self.sz, self.sz)),cmap = cm.binary, interpolation='nearest')
                self.ax[1,1].set_title('Iter=%d\nReconstruct (t)' % 0)

                self.ax[0,1].set_title('Reconstruction Error (t)')
                self.ax[0,1].set_xlabel('Time (steps)')
                self.ax[0,1].set_ylabel('SNR (dB)')
                self.ax[0,1].axis([0, self.num_frames * num_iterations, 0.0, 22])

                # Prediction
                p_recon = tdot(ZPhi, a)
                self.ax[1,0].imshow(np.reshape(p_recon, (self.sz, self.sz)),cmap = cm.binary, interpolation='nearest')
                self.ax[1,0].set_title('Iter=%d\nReconstruct (t+1)' % 0)

                self.ax[0,0].set_title('Reconstruction Error (t+1)')
                self.ax[0,0].set_xlabel('Time (steps)')
                self.ax[0,0].set_ylabel('SNR (dB)')
                self.ax[0,0].axis([0, self.num_frames * num_iterations, 0.0, 22])


                # The subplots move around if I don't do this lol...
                for i in range(6):
                    plt.savefig('animation/junk.png')
                if self.save_cgraphs:
                    plt.savefig('animation/%d.jpeg' % self.iter_idx)
                self.iter_idx += 1

            self.ax[2,1].imshow(np.reshape(I[:,0], (self.sz, self.sz)),cmap = cm.binary, interpolation='nearest')
            self.ax[2,1].set_title('Image (t)')

            self.ax[2,0].imshow(np.reshape(nI[:,0], (self.sz, self.sz)),cmap = cm.binary, interpolation='nearest')
            self.ax[2,0].set_title('Image (t+1)')

            if showme:
                plt.draw()
                plt.show()

        for t in range(num_iterations):
            #u = self.u_factor * (self.coeff_eta * (b + Zb - tdot(G,a) - tdot(ZG, a)) + (1 - self.coeff_eta) * u)
            #u = self.coeff_eta * 2.0 * (b - tdot(G,a)) + 2.0 * (1 - self.coeff_eta) * u
            u = self.coeff_eta * (b - tdot(G,a)) + (1 - self.coeff_eta) * u
            a = self.thresh(u, l)

            explode = check_activity_m(b, Zb, tdot(G, a), tdot(ZG, a), u)
            if explode:
                ahat_c = np.copy(a)
                ahat_c[np.abs(ahat_c) > self.lambdav/1000.0] = 1
                ac = np.sum(ahat_c)
                print 'Coefficients Active=%.2f%%' % (100 * float(ac)/(self.neurons * self.batch_size))


            l = self.lambda_decay * l
            l[l < self.lambdav] = self.lambdav

            if self.coeff_visualizer:
                ahat_c = np.copy(a)
                ahat_c[np.abs(ahat_c) > self.lambdav/1000.0] = 1
                ac = np.sum(ahat_c)
                self.ax[1,2].set_title('Coefficients Active=%.2f%%' % (100 * float(ac)/self.neurons))

                for coeff, i in zip(self.coeffs, range(self.neurons)):
                    coeff.set_height(abs(u[i]))  # Update the potentials
                self.lthresh[0].set_data(range(self.neurons+1), list(l) * (self.neurons+1))


                # Update Reconstruction
                recon = tdot(Phi, a)
                self.ax[1,1].imshow(np.reshape(recon, (self.sz, self.sz)),cmap = cm.binary, interpolation='nearest')
                self.ax[1,1].set_title('Iter=%d\nReconstruct (t)' % self.iter_idx)

                p_recon = tdot(ZPhi, a)
                self.ax[1,0].imshow(np.reshape(p_recon, (self.sz, self.sz)),cmap = cm.binary, interpolation='nearest')
                self.ax[1,0].set_title('Iter=%d\nReconstruct (t+1)' % self.iter_idx)

                # Plot SNR
                var = I.var().mean()
                R = I - recon
                mse = (R ** 2).mean()
                snr = 10 * log(var/mse, 10)
                color = 'r' if t == 0 else 'g'
                self.ax[0,1].scatter(self.iter_idx, snr, s=8, c=color)

                p_var = nI.var().mean()
                ZR = I - p_recon
                p_mse = (ZR ** 2).mean()
                p_snr = 10 * log(p_var/p_mse, 10)
                color = 'r' if t == 0 else 'g'
                self.ax[0,0].scatter(self.iter_idx, p_snr, s=8, c=color)

                if showme:
                    plt.draw()
                    plt.show()
                if self.save_cgraphs:
                    plt.savefig('animation/%d.jpeg' % self.iter_idx)
                self.iter_idx += 1

        return u, a

    def sparsify(self, I, Phi, u_pred=None, num_iterations=80):
        'Run the LCA coefficient dynamics'

        b = t2dot(Phi.T, I)
        G = t2dot(Phi.T, Phi) - np.eye(self.neurons)

        if self.fixed_lambda:
            l = np.ones(self.batch_size)
            l *= self.lambdav
            self.lambda_type = 'Fixed and lambdav'
        else:
            l = 0.5 * np.max(np.abs(b), axis = 0)
            self.lambda_type = 'l = 0.5 * np.max(np.abs(b), axis = 0)'

        u = u_pred if u_pred is not None else np.zeros((self.neurons, self.batch_size))
        a = self.thresh(u, l)

        showme = True # set to false if you just want to save the images
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

            for coeff, i in zip(self.coeffs, range(self.neurons)):
                coeff.set_height(abs(u[i]))  # Update the potentials
            self.lthresh[0].set_data(range(self.neurons+1), list(l) * (self.neurons+1))

            self.ax[1,0].imshow(np.reshape(I[:,0], (self.sz, self.sz)),cmap = cm.binary, interpolation='nearest')
            self.ax[1,0].set_title('Image')

            # Update Reconstruction
            recon = t2dot(Phi, a)
            self.ax[0,0].imshow(np.reshape(recon, (self.sz, self.sz)),cmap = cm.binary, interpolation='nearest')
            self.ax[0,0].set_title('Iter=%d\nReconstruct' % self.iter_idx)

            # Plot SNR
            var = I.var().mean()
            R = I - recon
            mse = (R ** 2).mean()
            snr = 10 * log(var/mse, 10)
            color = 'r'
            self.ax[0,1].scatter(self.iter_idx, snr, s=8, c=color)
            self.iter_idx += 1 # XXX Need to fix savfig + iter_idx

            if showme:
                plt.draw()
                plt.show()

        for t in range(num_iterations):
            u = self.coeff_eta * (b - t2dot(G,a)) + (1 - self.coeff_eta) * u
            a = self.thresh(u, l)

            check_activity(b, G, u, a)

            l = self.lambda_decay * l
            l[l < self.lambdav] = self.lambdav

            if self.coeff_visualizer:
                for coeff, i in zip(self.coeffs, range(self.neurons)):
                    coeff.set_height(abs(u[i]))  # Update the potentials
                self.lthresh[0].set_data(range(self.neurons+1), list(l) * (self.neurons+1))

                # Update Reconstruction
                recon = t2dot(Phi, a)
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
            if self.group_sparse > 1:
                a = self.group_thresh(u, theta)
            else:
                a[np.abs(a) < theta] = 0
        elif self.thresh_type=='soft': # L1 Approximation
            if self.group_sparse > 1:
                raise Exception("Group sparsity with soft threshold not supported")
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

    def view_log_save(self, Phi, comp, Z=None):
        'View Phi, Z, log parameters, and save the matrices '
        showbfs(Phi, self.phi_idx)
        plt.show()

        #if not self.log_and_save:
            #return

        if comp == 0.0: # Only log on first write
            self.phi_idx = self.phi_idx + 1
            name = 'Phi_%d' % self.phi_idx

            path = 'dict/%s' % name
            if not os.path.exists(path):
                os.makedirs(path)

            f = open('log.txt', 'a') # append to the log

            f.write('\n*** %s ***\n' % name)
            f.write('Time=%s\n' % datetime.now())
            f.write('RunType=%s\n' % get_RunType_name(self.runtype))
            f.write('IMAGES=%s\n' % self.image_data_name)
            f.write('patch_dim=%d\n' % self.patch_dim)
            f.write('neurons=%d\n' % self.neurons)
            f.write('lambdav=%.3f\n' % self.lambdav)
            f.write('Lambda Decay=%.2f\n' % self.lambda_decay)
            f.write('num_trials=%d\n' % self.num_trials)
            f.write('batch_size=%d\n' % self.batch_size)
            f.write('time_batch_size=%d\n' % self.time_batch_size)
            f.write('NUM_IMAGES=%d\n' % self.num_images)
            f.write('iter_per_frame=%d\n' % self.iters_per_frame)
            f.write('thresh_type=%s\n' % self.thresh_type)
            f.write('coeff_eta=%.3f\n' % self.coeff_eta)
            f.write('lambda_type=[%s]\n' % self.lambda_type)
            f.write('InitPhi=%s\n' % self.init_phi_name)
            f.write('Load Sequentially=%s\n' % self.load_sequentially)
            f.write('Skip initial frames=%s\n' % self.skip_frames)
            f.write('Group Sparse=%d\n' % self.group_sparse)

            f.write('%d\n' % (self.phi_idx))
            f.close()

            logfile = '%s/%s_log.txt' % (path, name)
            print 'Not Assigning stdout to %s' % logfile
            #print 'Assigning stdout to %s' % logfile
            #sys.stdout = open(logfile, 'w') # Rewire stdout to write the log

            # print these methods so we know the simulated annealing parameters
            print inspect.getsource(get_eta)
            print inspect.getsource(get_veta)
        else:
            name = 'Phi_%d' % self.phi_idx
            path = 'dict/%s' % name

        fname = '%s/%s_%.1f' % (path, name, comp)
        plt.savefig('%s.png' % fname)

        if Z is not None:
            scipy.io.savemat(fname, {'Phi':Phi, 'Z': Z})
            plt.imshow(Z, interpolation='nearest', norm=matplotlib.colors.Normalize(-1,1,True))
            plt.colorbar()
            plt.draw()
            plt.savefig('%s_Z.png' % fname)
            plt.show()
            plt.clf()
        else:
            scipy.io.savemat(fname, {'Phi':Phi})

        print '%s_%.1f successfully written.' % (name, comp)

    def run(self):
        if self.runtype == RunType.Learning:
            self.Learning()
        elif self.runtype == RunType.vLearning:
            self.vLearning()
        elif self.runtype == RunType.vmLearning:
            self.vmLearning()
        elif self.runtype == RunType.vReconstruct:
            self.vReconstruct()
        elif self.runtype == RunType.vPredict:
            self.vPredict()
        else:
            raise Exception("Unknown runtype specified")

lca = LcaNetwork()
lca.run()
