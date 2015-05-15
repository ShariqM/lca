import matplotlib
import scipy.io
import scipy.stats as stats
import time
import numpy as np
from numpy import reshape, zeros, ones
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from math import floor, ceil, sqrt
import pprint
import argparse
import pdb



class Analysis():

    def __init__(self, data):
        self.activity_log = np.load(data)
        self.nframes = self.activity_log.shape[2]
        self.patches = self.activity_log.shape[1]


    def over_time(self, coeffs, patch_index):
        pdb.set_trace()
        for pi in range(3): #(0,1,2,3):
            for i in coeffs:
                plt.plot(range(self.nframes), self.activity_log[i, pi,:], label='P_%d_A%d' % (pi, i))
        plt.title("Activity over %d frames for patch %d" % (self.nframes, patch_index))
        #plt.legend()
        plt.show()

    def spatial_correlation(self, coeffs, patch_index):
        if len(coeffs) != 2:
            raise Exception("Greater than 2 dim not supported.")

        patches = range(self.patches)[:10]
        x = self.activity_log[coeffs[0], patches, :]
        y = self.activity_log[coeffs[1], patches, :]
        #for i in range(self.patches):
            #pdb.set_trace()
            #x += self.activity_log[coeffs[0], i, :]
            #y += self.activity_log[coeffs[1], i, :]

        h = max(np.max(x), np.max(y))
        plt.plot([-h, h], [0, 0], color='k')
        plt.plot([0, 0], [-h, h], color='k')

        plt.scatter(x, y, s=1, color='red')
        plt.xlabel("A%d" % coeffs[0], fontdict={'fontsize':18})
        plt.ylabel("A%d" % coeffs[1], fontdict={'fontsize':18})
        plt.axis('equal')
        plt.show()

    def temporal_correlation(self, coeffs, patch_index):
        if len(coeffs) != 2:
            raise Exception("Greater than 2 dim not supported.")

        patches = range(self.patches)[:10]
        x = self.activity_log[coeffs[0], patches, :]
        y = self.activity_log[coeffs[1], patches, :]
        #for i in range(self.patches):
            #pdb.set_trace()
            #x += self.activity_log[coeffs[0], i, :]
            #y += self.activity_log[coeffs[1], i, :]

        h = max(np.max(x), np.max(y))
        plt.plot([-h, h], [0, 0], color='k')
        plt.plot([0, 0], [-h, h], color='k')

        plt.scatter(x, y, s=1, color='red')
        plt.xlabel("A%d" % coeffs[0], fontdict={'fontsize':18})
        plt.ylabel("A%d" % coeffs[1], fontdict={'fontsize':18})
        plt.axis('equal')
        plt.show()

patch_index = 8
#data = 'activity_Phi_520_0.6.npy'
#group = [0,1,2,3]

data = 'activity_Phi_524_0.4.npy'
#group = [0,1]
group = [46,47] # Strong correlatioN
group = [0,1]

a = Analysis(data)
a.over_time(group, patch_index)
#a.spatial_correlation(group, patch_index)
