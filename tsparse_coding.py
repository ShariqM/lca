from theano import *
from helpers import *
import theano.tensor as T
from datetime import datetime
import time
import pdb
import random

import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
from showbfs import showbfs
import scipy.io
import numpy as np
from math import log

if socket.gethostname() == 'redwood2':
    matplotlib.use('Agg') # Don't crash because $Display is not set correctly on the cluster

class SparseCoding():

    patch_dim = 144
    neurons = 72
    #neurons = patch_dim
    batch_size = 100
    time_batch_size = 100
    border = 4
    sz = np.sqrt(patch_dim)
    omega = 0.5    # Reconstruction Penalty
    lambdav = 0.35 # Sparsity penalty
    gamma = 0.01    # Coeff Prediction Penalty
    kappa = 0.01    # Smoothness penalty

    num_trials = 1000
    coeff_iterations = 100

    visualize = False
    initialized = False

    def __init__(self, obj):
        image_data_name = 'IMAGES_MOVE_RIGHT'
        #image_data_name = 'IMAGES_EDGE_RIGHT_DUCK'
        #image_data_name = 'IMAGES_DUCK_SHORT' # XXX CHANGE TO DUCK
        self.IMAGES = scipy.io.loadmat('mat/%s.mat' % image_data_name)[image_data_name]
        (self.imsize, imsize, self.num_images) = np.shape(self.IMAGES)
        self.obj = obj

        if self.visualize:
            plt.ion()
            self.batch_size = 1

        I = T.fmatrix('I') # Image
        D = T.fmatrix('D') # Dictionary
        A = T.fmatrix('A') # Coefficients
        l = T.dscalar('l') # Lambda (sparse penalty)

        if self.obj == 1:
            E = 0.5 * ((I - T.dot(D, A)).norm(2) ** 2) + l * A.norm(1)

            self.fE = function([I, D, A, l], E, allow_input_downcast=True)

            ga = T.grad(E, A)
            gD = T.grad(E, D)

            self.fgA = function([I, D, A, l], ga, allow_input_downcast=True)
            self.fgD = function([I, D, A, l], gD, allow_input_downcast=True)
        else:

            # Messy, need to get an initial A_past
            sc_E = 0.5 * ((I - T.dot(D, A)).norm(2) ** 2) + l * A.norm(1)
            self.sc_fE = function([I, D, A, l], sc_E, allow_input_downcast=True)
            sc_ga = T.grad(sc_E, A)
            self.sc_fgA = function([I, D, A, l], sc_ga, allow_input_downcast=True)


            A_past = T.fmatrix('A_past')
            Z = T.fmatrix('Z')
            o = T.dscalar('o') # Omega (Reconstruction Penalty)
            g = T.dscalar('g') # Gamma (Coeff Prediction Penalty)
            k = T.dscalar('k') # Kappa (Smoothness Penalty)

            # Correct
            #E = 0.5 * ((I - T.dot(D, A)).norm(2) ** 2) + l * A.norm(1) + g * (A - T.dot(Z, A_past)).norm(2) + k * (A - A_past).norm(2)

            # Works
            #E = 0.5 * ((I - T.dot(D, A)).norm(2) ** 2) + l * A.norm(1) + .0001 * g * k * Z.norm(1) * A_past.norm(1)

            # Doesn't work with l2
            #E = 0.5 * ((I - T.dot(D, A)).norm(2) ** 2) + l * A.norm(1) + g * (A - T.dot(Z, A_past)).norm(1) + .0001 * g * k * Z.norm(1) * A_past.norm(1)
            #E = 0.5 * ((I - T.dot(D, A)).norm(2) ** 2) + l * A.norm(1) + k * (A - A_past).norm(1) + .0001 * g * k * Z.norm(1) * A_past.norm(1)

            #E = 0.5 * ((I - T.dot(D, A)).norm(2) ** 2) + l * A.norm(1) + g * (A - T.dot(Z, A_past)).norm(2) + .0001 * g * k * Z.norm(1) * A_past.norm(1) # Nan

            E = o * ((I - T.dot(D, A)).norm(2) ** 2) + l * A.norm(1)
            # L2 doesn't work
            E += g * (A - T.dot(Z, A_past)).norm(1) # Prediction Error
            E += k * (A - A_past).norm(1) # Smoothness penalty

            self.fE = function([I, D, Z, A, A_past, o, l, g, k], E, allow_input_downcast=True)

            ga = T.grad(E, A)
            gD = T.grad(E, D)
            gZ = T.grad(E, Z)

            self.fgA = function([I, D, Z, A, A_past, o, l, g, k], ga, allow_input_downcast=True)
            self.fgD = function([I, D, Z, A, A_past, o, l, g, k], gD, allow_input_downcast=True)
            self.fgZ = function([I, D, Z, A, A_past, o, l, g, k], gZ, allow_input_downcast=True)

    # Image Loaders
    def load_rimages(self, I):
        '(1) Choose A batch_size number of random images. Used by Learning()'

        border, imsize, sz = self.border, self.imsize, self.sz
        imi = np.ceil(self.num_images * random.uniform(0, 1))
        # Pick batch_size random patches from the random image
        for i in range(self.batch_size):
            r = border + np.ceil((imsize-sz-2*border) * random.uniform(0, 1))
            c = border + np.ceil((imsize-sz-2*border) * random.uniform(0, 1))
            I[:,i] = np.reshape(self.IMAGES[r:r+sz, c:c+sz, imi-1], self.patch_dim, 1)

        return I

    def load_videos(self):
        '(3) Load a batch_size number of Space-Time boxes of individual random patches. Used by vLearning()'

        border, imsize, sz, tbs = self.border, self.imsize, self.sz, self.time_batch_size
        VI = np.zeros((self.patch_dim, self.batch_size, self.time_batch_size))
        for x in range(self.batch_size):
            # Choose a random image less than time_batch_size images away from the end
            imi = np.floor((self.num_images - self.time_batch_size) * random.uniform(0, 1))
            r = border + np.ceil((imsize-sz-2*border) * random.uniform(0, 1))
            c = border + np.ceil((imsize-sz-2*border) * random.uniform(0, 1))
            VI[:,x,:] = np.reshape(self.IMAGES[r:r+sz, c:c+sz, imi:imi+tbs], (self.patch_dim, tbs), 1)
        return VI

    def scLearning(self):
        D = np.random.randn(self.patch_dim, self.neurons)

        # Initialize batch of images
        I = np.zeros((self.patch_dim, self.batch_size))

        start = datetime.now()
        for t in range(0, self.num_trials):
            I = self.load_rimages(I)
            #showbfs(I, 111)
            #plt.show()

            A = self.sparsify(I, D)

            print '%.3d) Error_1 = %d' % (t, self.fE(I, D, A, self.lambdav))
            eta = 0.5 * get_eta(t, self.neurons, -1, self.batch_size)
            D = D - eta * self.fgD(I, D, A, self.lambdav)
            print '%.3d) Error_2 = %d' % (t, self.fE(I, D, A, self.lambdav))
            D = t2dot(D, np.diag(1/np.sqrt(np.sum(D**2, axis = 0))))

            if t % 200 == 0:
                showbfs(D, -1)
                plt.show()

    def sparsify(self, I, D):
        coeff_eta = 0.0050
        A = np.zeros((self.neurons, self.batch_size))
        for i in range(self.coeff_iterations):
                grad = self.fgA(I, D, A, self.lambdav)
                A = A - coeff_eta * grad
                print '%.3d) Error= %d' % (i, self.fE(I, D, A, self.lambdav))

        thresh = 0.05
        G = np.copy(A)
        G[G > thresh] = 1
        G[G <= thresh] = 0

        var = I.var().mean()
        recon = tdot(D, G)
        R = I - recon
        mse = (R ** 2).mean()
        snr = 10 * log(var/mse, 10)
        print '%.3d) SNR=%.2fdB, Activity=%.2f%%' % (0, snr, np.sum(G)/self.batch_size)

        if self.visualize:
            if not self.initialized:
                fg, self.ax = plt.subplots(2,1)
                self.initialized = True

            self.ax[0].imshow(np.reshape(I[:,0], (self.sz, self.sz)),cmap = cm.binary, interpolation='nearest')
            self.ax[0].set_title('Image')

            self.ax[1].imshow(np.reshape(recon, (self.sz, self.sz)),cmap = cm.binary, interpolation='nearest')
            self.ax[1].set_title('Reconstruct')
            plt.draw()
            plt.show()

        return A

    def showZ(self, Z):
        plt.imshow(Z, interpolation='nearest', norm=matplotlib.colors.Normalize(-1,1,True))
        plt.colorbar()
        plt.show()

    def mscLearning(self):
        D = np.random.randn(self.patch_dim, self.neurons)
        Z = np.eye(self.neurons)
        self.showZ(Z)

        old_coeff_eta = 0.0025
        coeff_eta     = 0.01000
        for t in range(0, self.num_trials):
            VI = self.load_videos()
            I = VI[:,:,0]

            A      = np.zeros((self.neurons, self.batch_size))
            A_past = np.zeros((self.neurons, self.batch_size))
            # Have to get an A_past loaded up before learning
            print '%.3d) Old_Error_0 = %d' % (t, self.sc_fE(I, D, A_past, self.lambdav))
            for i in range(self.coeff_iterations):
                grad = self.sc_fgA(I, D, A_past, self.lambdav)
                #print 'A_past grad', grad
                A_past = A_past - old_coeff_eta * grad
            print '%.3d) Old_Error_1 = %d' % (t, self.sc_fE(I, D, A_past, self.lambdav))
            A = A_past

            for q in range(1, self.time_batch_size):
                I = VI[:,:,q]

                print '%.3d) Error_2 = %d' % (t, self.fE(I, D, Z, A, A_past, self.omega, self.lambdav, self.gamma, self.kappa))
                for i in range(self.coeff_iterations):
                    grad = self.fgA(I, D, Z, A, A_past, self.omega, self.lambdav, self.gamma, self.kappa)
                    #print 'A', grad
                    A = A - coeff_eta * grad

                var = I.var().mean()
                R = I - tdot(D, A)
                p_R = I - t3dot(D, Z, A_past)
                mse = (R ** 2).mean()
                p_mse = (p_R ** 2).mean()
                snr = 10 * log(var/mse, 10)
                p_snr = 10 * log(var/p_mse, 10)
                print '%.3d) SNR=%.2fdB P_SNR=%.2fdB' % (t, snr, p_snr)

                print '%.3d) Error_3 = %d' % (t, self.fE(I, D, Z, A, A_past, self.omega, self.lambdav, self.gamma, self.kappa))
                eta = get_zeta(self.batch_size * t, self.neurons, -1, self.batch_size)
                Z = Z - eta * self.fgZ(I, D, Z, A, A_past, self.omega, self.lambdav, self.gamma, self.kappa)
                print '%.3d) Error_4 = %d' % (t, self.fE(I, D, Z, A, A_past, self.omega, self.lambdav, self.gamma, self.kappa))

                eta = get_veta(self.batch_size * t, self.neurons, -1, self.batch_size)
                D = D - eta * self.fgD(I, D, Z, A, A_past, self.omega, self.lambdav, self.gamma, self.kappa)
                D = t2dot(D, np.diag(1/np.sqrt(np.sum(D**2, axis = 0))))
                print '%.3d) Error_5 = %d' % (t, self.fE(I, D, Z, A, A_past, self.omega, self.lambdav, self.gamma, self.kappa))
                print '\n'

                A_past = A

            #if t % 20 == 0:
            if t % 5 == 0:
                print '%.3d) Activity for %d neurons is %f' % (t, self.neurons, np.sum(np.abs(A)))
                print '%.3d) Error_1 = %d\n' % (t, self.fE(I, D, Z, A, A_past, self.omega, self.lambdav, self.gamma, self.kappa))
                showbfs(D, -1)
                plt.show()
                self.showZ(Z)


        scipy.io.savemat('Dict_1', {'D':D, 'Z': Z})



    def run(self):
        if self.obj == 1:
            self.scLearning()
        else:
            self.mscLearning()

sc = SparseCoding(1)
sc.run()
