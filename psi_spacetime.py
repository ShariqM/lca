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
import pdb
from helpers import *
import sys

dtype = theano.config.floatX
class SpaceTime():

    a_mode    = True
    phi_name  = 'Phi_463_0.3'
    data_name = 'IMAGES_DUCK_SHORT'

    time_batch_size = 64
    cells = 200
    timepoints = 7

    def __init__(self):
        'Log of ahat or u values'
        aname = 'activity' if self.a_mode else 'membrane'
        self.log = np.load('activity/%s_%s_%s.npy' % (aname, self.phi_name, self.data_name))

        self.neurons = self.log.shape[0]
        self.batch_size = self.log.shape[1] # batch_size
        self.nframes = self.log.shape[2]


    def sparsify():

    def train():
        Psi = np.zeros((self.neurons, self.cells, self.timepoints))

        for t in range(time_batch_size):
            a = np.zeros((self.cells, self.time_batch_size))








