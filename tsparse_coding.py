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
#from Recurrnet import recurrnet
#from recurrnet import optimizer
#from optimizer import *
from Recurrnet.recurrnet.optimizer import *
from objectivetype import *

dtype = theano.config.floatX

class SparseCoding():

    patch_dim = 144
    neurons = 144
    #neurons = 2 * patch_dim
    batch_size = 100
    time_batch_size = 100
    border = 4
    sz = np.sqrt(patch_dim)
    omega = 0.25    # Reconstruction Penalty
    lambdav = 0.044 # Sparsity penalty
    #lambdav = 1.00 # Sparsity penalty
    gamma = 8.00    # Coeff Prediction Penalty
    kappa =  4.00    # Smoothness penalty

    obj = ObjectiveType.BSC

    art_thresh = 0.05

    num_trials = 10000
    coeff_iterations = 100

    visualize = False
    initialized = False

    #def t_threshold(self, A):
        #g = -5
        #return A / (1 + T.exp(g * (A - 1))

    def __init__(self):
        #image_data_name = 'IMAGES_MOVE_RIGHT'
        #image_data_name = 'IMAGES_DUCK'
        image_data_name = 'IMAGES_DUCK_SHORT'
        self.IMAGES = scipy.io.loadmat('mat/%s.mat' % image_data_name)[image_data_name]
        (self.imsize, imsize, self.num_images) = np.shape(self.IMAGES)

        self.last_G = None
        if self.visualize:
            plt.ion()
            self.batch_size = 1

        I = T.fmatrix('I') # Image

        Dm = np.random.randn(self.patch_dim, self.neurons)
        D = self.D = theano.shared(Dm.astype(dtype))

        Am = np.zeros((self.neurons, self.batch_size))
        A = self.A = theano.shared(Am.astype(dtype))

        l = T.dscalar('l') # Lambda (sparse penalty)

        if self.obj == ObjectiveType.BSC:
            E = 0.5 * ((I - T.dot(D, T.exp(1+A))).norm(2) ** 2) + l
            self.fE = function([I, l], E, allow_input_downcast=True)
            params = [D, A]
            gparams = T.grad(E, wrt=params)
            updates = adadelta_update(params, gparams)
            print 'a'

            self.learn_D = theano.function(inputs = [I, l],
                                    outputs = E,
                                    updates = [[D, updates[D]]],
                                    allow_input_downcast=True)

            self.learn_A = theano.function(inputs = [I, l],
                                    outputs = E,
                                    updates = [[A, updates[A]]],
                                    allow_input_downcast=True) # Brian doesn't seem to use this


        elif self.obj == ObjectiveType.SC or self.obj == ObjectiveType.VSC:
            #E = 0.5 * ((I - T.dot(D, T.clip(A, l, 1000) - l)).norm(2) ** 2) + l * A.norm(1)
            E = 0.5 * ((I - T.dot(D, A)).norm(2) ** 2) + l * A.norm(1)
            #E = 0.5 * ((I - T.dot(D, A)).norm(2) ** 2) + l * T.sum(T.log(1 + (A/0.1) ** 2)) # Set L=0.04
            self.fE = function([I, l], E, allow_input_downcast=True)
            params = [D, A]
            gparams = T.grad(E, wrt=params)
            updates = adadelta_update(params, gparams)

            self.learn_D = theano.function(inputs = [I, l],
                                    outputs = E,
                                    updates = [[D, updates[D]]],
                                    allow_input_downcast=True)

            self.learn_A = theano.function(inputs = [I, l],
                                    outputs = E,
                                    updates = [[A, updates[A]]],
                                    allow_input_downcast=True) # Brian doesn't seem to use this
        elif self.obj == ObjectiveType.GSC:
            Gm = initG(self.neurons)
            self.showZ(Gm)
            G = theano.shared(Gm.astype(dtype))

            E = 0.5 * ((I - T.dot(D, A)).norm(2) ** 2) + l * T.dot(G, A).norm(1)
            self.fE = function([I, l], E, allow_input_downcast=True)
            params = [D, A]
            gparams = T.grad(E, wrt=params)
            updates = adadelta_update(params, gparams)

            self.learn_D = theano.function(inputs = [I, l],
                                    outputs = E,
                                    updates = [[D, updates[D]]],
                                    allow_input_downcast=True)

            self.learn_A = theano.function(inputs = [I, l],
                                    outputs = E,
                                    updates = [[A, updates[A]]],
                                    allow_input_downcast=True) # Brian doesn't seem to use this
        elif self.obj == ObjectiveType.PPSC:
            A_initm = np.zeros((self.neurons, self.batch_size))
            A_init = self.A_init = theano.shared(A_initm.astype(dtype))

            # Messy have to do this to init A_past
            sc_E = 0.5 * ((I - T.dot(D, A_init)).norm(2) ** 2) + l * A_init.norm(1)
            self.sc_fE = function([I, l], sc_E, allow_input_downcast=True)
            params = [A_init]
            gparams = T.grad(sc_E, wrt=params)
            updates = adadelta_update(params, gparams)

            self.learn_A_init = theano.function(inputs = [I, l],
                                    outputs = sc_E,
                                    updates = [[A_init, updates[A_init]]],
                                    allow_input_downcast=True)

            A_past = T.fmatrix('A_past')
            k = T.dscalar('k') # Kappa (Smoothness Penalty)
            E = 0.5 * ((I - T.dot(D, A)).norm(2) ** 2) + l * A.norm(1) + k * (A ** 2 - A_past ** 2).norm(1)
            self.fE = function([I, A_past, l, k], E, allow_input_downcast=True)
            params = [D, A]
            gparams = T.grad(E, wrt=params)
            updates = adadelta_update(params, gparams)

            self.learn_D = theano.function(inputs = [I, A_past, l, k],
                                    outputs = E,
                                    updates = [[D, updates[D]]],
                                    allow_input_downcast=True)

            self.learn_A = theano.function(inputs = [I, A_past, l, k],
                                    outputs = E,
                                    updates = [[A, updates[A]]],
                                    allow_input_downcast=True)

            Zm = np.eye(self.neurons)
            Z = self.Z = theano.shared(Zm.astype(dtype))
            #o = T.dscalar('o') # Omega (Reconstruction Penalty)
            g = T.dscalar('g') # Gamma (Coeff Prediction Penalty)

            E = g * (A ** 2 - T.dot(Z, A_past) ** 2).norm(1)
            self.fE = function([A_past, g], E, allow_input_downcast=True)

            params = [Z]
            gparams = T.grad(E, wrt=params)
            updates = adadelta_update(params, gparams)

            self.learn_Z = theano.function(inputs = [A_past, g],
                                    outputs = E,
                                    updates = [[Z, updates[Z]]],
                                    allow_input_downcast=True)
        elif self.obj == ObjectiveType.PSC:
            A_initm = np.zeros((self.neurons, self.batch_size))
            A_init = self.A_init = theano.shared(A_initm.astype(dtype))

            # Messy have to do this to init A_past
            sc_E = 0.5 * ((I - T.dot(D, A_init)).norm(2) ** 2) + l * A_init.norm(1)
            self.sc_fE = function([I, l], sc_E, allow_input_downcast=True)
            params = [A_init]
            gparams = T.grad(sc_E, wrt=params)
            updates = adadelta_update(params, gparams)

            self.learn_A_init = theano.function(inputs = [I, l],
                                    outputs = sc_E,
                                    updates = [[A_init, updates[A_init]]],
                                    allow_input_downcast=True)

            A_past = T.fmatrix('A_past')
            Zm = np.eye(self.neurons)
            Z = self.Z = theano.shared(Zm.astype(dtype))

            o = T.dscalar('o') # Omega (Reconstruction Penalty)
            g = T.dscalar('g') # Gamma (Coeff Prediction Penalty)
            k = T.dscalar('k') # Kappa (Smoothness Penalty)

            # Correct
            E = o * ((I - T.dot(D, A)).norm(2) ** 2) + l * A.norm(1)
            # L2 doesn't work
            #E += g * (A ** 2 - T.dot(Z, A_past) ** 2).norm(1) # Prediction Error
            E += g * (A - T.dot(Z, A_past)).norm(1) # Prediction Error
            E += k * (A ** 2 - A_past ** 2).norm(1) # Smoothness penalty
            #E += k * (A - A_past).norm(1) # Smoothness penalty

            self.fE = function([I, A_past, o, l, g, k], E, allow_input_downcast=True)

            params = [D, A, Z]
            gparams = T.grad(E, wrt=params)
            updates = adadelta_update(params, gparams)

            self.learn_D = theano.function(inputs = [I, A_past, o, l, g, k],
                                    outputs = E,
                                    updates = [[D, updates[D]]],
                                    allow_input_downcast=True)
            self.learn_A = theano.function(inputs = [I, A_past, o, l, g, k],
                                    outputs = E,
                                    updates = [[A, updates[A]]],
                                    allow_input_downcast=True)

            self.learn_Z = theano.function(inputs = [I, A_past, o, l, g, k],
                                    outputs = E,
                                    updates = [[Z, updates[Z]]],
                                    allow_input_downcast=True)

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

    def log_snr_sparsity(self, t, I, start):
        A = self.A.get_value()
        D = self.D.get_value()

        thresh = self.art_thresh
        G = np.copy(A)
        G[np.abs(G) > thresh] = 1
        G[np.abs(G) <= thresh] = 0

        var = I.var().mean()
        recon = tdot(D, A)
        R = I - recon
        mse = (R ** 2).mean()
        snr = 10 * log(var/mse, 10)
        print '%.3d) SNR=%.2fdB, Activity=%.2f%%, Time=%ds' % \
                (t, snr, np.sum(G)/self.batch_size, (datetime.now() - start).seconds)

    def log_psnr_sparsity(self, t, I, A_past, start):
        A = self.A.get_value()
        D = self.D.get_value()
        Z = self.Z.get_value()

        # Sparsity
        thresh = self.art_thresh
        G = np.copy(A)
        G[np.abs(G) > thresh] = 1
        G[np.abs(G) <= thresh] = 0

        AC = np.sum(G)/self.batch_size
        CC_SUM = np.sum(np.abs(np.abs(A)-np.abs(A_past)))
        AC_SUM = np.sum(np.abs(A))
        #if self.last_G != None:
            #CC = np.sum(self.last_G!=G)
            #self.last_A = np.copy(A)
        #else:
            #CC = 0

        # SNR, P_SNR, I_SNR
        var = I.var().mean()

        R   = I - t2dot(D, A)
        p_R = I - t2dot(D, t2dot(Z, A_past))
        i_R = I - t2dot(D, A_past)

        mse   = (R ** 2).mean()
        p_mse = (p_R ** 2).mean()
        i_mse = (i_R ** 2).mean()

        snr   = 10 * log(var/mse, 10)
        p_snr = 10 * log(var/p_mse, 10)
        i_snr = 10 * log(var/i_mse, 10)

        print '%.3d) SNR=%.2fdB, I_SNR=%.2fdB, P_SNR=%.2fdB, AC=%.2f%%, AC_SUM=%f, CC_SUM=%f, Time=%ds' %  \
                (t, snr, i_snr, p_snr, AC, AC_SUM, CC_SUM, (datetime.now() - start).seconds)

    def scLearning(self):
        # Initialize batch of images
        print 'learn'
        I = np.zeros((self.patch_dim, self.batch_size))

        start = datetime.now()
        for t in range(0, self.num_trials):
            I = self.load_rimages(I)

            self.sparsify(t, I)
            self.learn_D(I, self.lambdav)
            D = self.D.get_value()
            self.D.set_value(t2dot(D, np.diag(1/np.sqrt(np.sum(D**2, axis = 0)))))

            if t % 80 == 0:
                self.log_snr_sparsity(t, I, start)
                showbfs(self.D.get_value(), -1)
                plt.show()

    def threshold_A(self, A):
        thresh = self.art_thresh

        G = np.copy(A)
        G = np.abs(G) - thresh
        G[G < 0] = 0
        self.A.set_value(np.sign(A) * G)

        #G = np.copy(A)
        #G[np.abs(G) < thresh] = 0
        #self.A.set_value(G)

    def vscLearning(self):
        start = datetime.now()
        for t in range(0, self.num_trials):
            VI = self.load_videos()

            for i in range(self.time_batch_size):
                I = VI[:,:,i]
                self.sparsify(t, I, reinit=False)

                A = self.A.get_value()
                self.threshold_A(A) # Gradient only for bigger A
                self.learn_D(I, self.lambdav)
                self.A.set_value(A)

                D = self.D.get_value()
                self.D.set_value(t2dot(D, np.diag(1/np.sqrt(np.sum(D**2, axis = 0)))))

            if t % 5 == 0:
                self.log_snr_sparsity(t, I, start)
                showbfs(self.D.get_value(), -1)
                plt.show()

    def sparsify(self, t, I, reinit=True):
        if reinit:
            self.A.set_value(np.zeros((self.neurons, self.batch_size)).astype(dtype))
        for i in range(self.coeff_iterations):
            self.learn_A(I, self.lambdav)

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

    def showZ(self, Z):
        plt.imshow(Z, interpolation='nearest', norm=matplotlib.colors.Normalize(-1,1,True))
        plt.colorbar()
        plt.show()

    def ppscLearning(self):
        start = datetime.now()
        for t in range(0, self.num_trials):
            VI = self.load_videos()
            I = VI[:,:,0]

            # Have to get an A_past loaded up before learning
            for i in range(self.coeff_iterations):
                self.learn_A_init(I, self.lambdav)
            A = self.A_init.get_value()

            for q in range(1, self.time_batch_size):
                A_past = self.A.get_value()
                I = VI[:,:,q]

                for i in range(self.coeff_iterations):
                    self.learn_A(I, A_past, self.lambdav, self.kappa)

                self.learn_Z(A_past, self.gamma)
                if t % 10 == 0: # Update slower
                    self.learn_D(I, A_past, self.lambdav, self.kappa)


            if t % 5 == 0:
                self.log_psnr_sparsity(t, I, A_past, start)
                showbfs(self.D.get_value(), -1)
                plt.show()
                self.showZ(self.Z.get_value())
                name = 'sc_dict/Dict_%s' % start
                scipy.io.savemat(name, {'D':self.D.get_value(), 'Z': self.Z.get_value()})


    def pscLearning(self):
        start = datetime.now()
        for t in range(0, self.num_trials):
            VI = self.load_videos()
            I = VI[:,:,0]

            # Have to get an A_past loaded up before learning
            for i in range(self.coeff_iterations):
                self.learn_A_init(I, self.lambdav)
            A = self.A_init.get_value()

            for q in range(1, self.time_batch_size):
                A_past = self.A.get_value()
                I = VI[:,:,q]

                #print '%.3d) Error_2 = %d' % (t, self.fE(I, A_past, self.omega, self.lambdav, self.gamma, self.kappa))
                for i in range(self.coeff_iterations):
                    self.learn_A(I, A_past, self.omega, self.lambdav, self.gamma, self.kappa)

                self.learn_Z(I, A_past, self.omega, self.lambdav, self.gamma, self.kappa)
                #if t % 10 == 0: # Update slower
                self.learn_D(I, A_past, self.omega, self.lambdav, self.gamma, self.kappa)


            if t % 5 == 0:
                self.log_psnr_sparsity(t, I, A_past, start)
                print '%.3d) psc - Error_1 = %d\n' % (t, self.fE(I, A_past, self.omega, self.lambdav, self.gamma, self.kappa))
                showbfs(self.D.get_value(), -1)
                plt.show()
                self.showZ(self.Z.get_value())
                name = 'sc_dict/Dict_%s' % start
                scipy.io.savemat(name, {'D':self.D.get_value(), 'Z': self.Z.get_value()})


    def run(self):
        if self.obj == ObjectiveType.BSC or self.obj == ObjectiveType.SC or self.obj == ObjectiveType.GSC:
            self.scLearning()
        elif self.obj == ObjectiveType.VSC:
            self.vscLearning()
        elif self.obj == ObjectiveType.PPSC:
            self.ppscLearning()
        elif self.obj == ObjectiveType.PSC:
            self.pscLearning()
        else:
            raise Exception("Unknown objective type")

sc = SparseCoding()
sc.run()
