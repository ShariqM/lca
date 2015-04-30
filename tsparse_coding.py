from theano import *
from helpers import *
import theano.tensor as T
from datetime import datetime
import time
import pdb
import random

import matplotlib.pyplot as plt
from showbfs import showbfs
import scipy.io
import numpy as np

class SparseCoding():

    patch_dim = 144
    neurons = patch_dim * 1
    batch_size = 100
    border = 4
    sz = np.sqrt(patch_dim)
    lambdav = 0.25

    num_trials = 1000

    def __init__(self):

        image_data_name = 'IMAGES_DUCK_SHORT'
        self.IMAGES = scipy.io.loadmat('mat/%s.mat' % image_data_name)[image_data_name]
        (self.imsize, imsize, self.num_images) = np.shape(self.IMAGES)

        I = T.fmatrix('I') # Image
        D = T.fmatrix('D') # Dictionary
        a = T.fmatrix('a') # Coefficients
        L = T.dscalar('L') # Lambda

        E = (1.0/2.0) * ((I - T.dot(D, a)).norm(2) ** 2) + L * a.norm(1)

        self.fE = function([I, D, a, L], E, allow_input_downcast=True)

        ga = T.grad(E, a)
        gD = T.grad(E, D)

        #print pp(ga)
        #print pp(gD)

        self.fga = function([I, D, a, L], ga, allow_input_downcast=True)
        self.fgD = function([I, D, a, L], gD, allow_input_downcast=True)


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

    def Learning(self):
        print 'Learning'
        D = np.random.randn(self.patch_dim, self.neurons)

        # Initialize batch of images
        I = np.zeros((self.patch_dim, self.batch_size))
        a = np.zeros((self.neurons, self.batch_size))

        coeff_eta = 0.0025
        start = datetime.now()
        for t in range(0, self.num_trials):
            I = self.load_rimages(I)

            for i in range(60):
                grad = self.fga(I, D, a, self.lambdav)
                a = a - coeff_eta * grad
                #print '%.3d) Error= %d' % (i, self.fE(I, D, a, self.lambdav))

            print 'Activity for %d neurons is %d' % (self.neurons, np.sum(np.abs(a)))
            print '%.3d) Error_1 = %d' % (t, self.fE(I, D, a, self.lambdav))
            eta = 1 * get_eta(t, self.neurons, -1, self.batch_size)
            D = D - eta * self.fgD(I, D, a, self.lambdav)
            print '%.3d) Error_2 = %d' % (t, self.fE(I, D, a, self.lambdav))
            D = t2dot(D, np.diag(1/np.sqrt(np.sum(D**2, axis = 0))))

            if t % 20 == 0:
                showbfs(D, -1)
                plt.show()


    def run(self):
        self.Learning()

sc = SparseCoding()
sc.run()
