import matplotlib
import math
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
        tstart = 560
        tend = 580

        for pi in range(1): #(0,1,2,3):
            pi = 1
            for i in coeffs:
                #plt.plot(range(self.nframes), self.activity_log[i, pi,:], label='P_%d_A%d' % (pi, i))
                plt.plot(range(tend - tstart), self.activity_log[i, pi, tstart:tend], label='P_%d_A%d' % (pi, i))
        plt.title("Activity over %d frames for patch %d" % (self.nframes, patch_index))
        plt.legend()
        plt.show()

    def spatial_correlation(self, coeffs, patch_index, log=False):
        if len(coeffs) != 2:
            raise Exception("Greater than 2 dim not supported.")

        patches = range(self.patches)[:3]
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

    def train_dynamics(self):
        tstart = 569
        tend = 575
        coeffs = [0,1]
        pi = 1
        data = self.activity_log[coeffs, pi, tstart:tend]
        Z = np.random.randn(2, 2)
        Z = np.eye(2)
        eta = 2.0

        # Obj a - Za
        for t in range(10000):
            print 'T=%d' % t
            for i in range(1, tend - tstart):
                a_prev, a = (data[:, i-1], data[:, i])
                R = a - np.dot(Z, a_prev)
                print '\ti=%d, R1=%f' % (i, np.sqrt(np.sum(np.abs(R))))
                Z = Z + eta * np.dot(R, a_prev.T)
                R2 = a - np.dot(Z, a_prev)
                print '\ti=%d, R2=%f' % (i, np.sqrt(np.sum(np.abs(R2))))
                print ''
        print 'Z', Z

    def temporal_correlation(self, coeffs, time, patch_index):
        if len(coeffs) != 2:
            raise Exception("Greater than 2 dim not supported.")

        tstart, tend = time[0], time[1]
        pi = patch_index
        data = self.activity_log[coeffs, pi, tstart:tend]

        cartesian = False
        if True:
            fg, ax = plt.subplots(3,1)
            h = np.max(np.abs(data))
            ax[0].plot([-h, h], [0, 0], color='k')
            ax[0].plot([0, 0], [-h, h], color='k')

            colors = np.linspace(0, 1, tend - tstart)
            cmap = plt.get_cmap('autumn')

            #scat = plt.scatter(data[0], data[1], s=12, c=colors, cmap=cmap, lw=0)
            pdb.set_trace()
            scat = ax[0].scatter(data[0], data[1], s=20, c=colors, lw=1)
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

patch_index = 8
#data = 'activity_Phi_520_0.6.npy'
#group = [0,1,2,3]

data = 'activity_Phi_524_0.4.npy'
#group = [0,1]
group = [46,47] # Strong correlatioN
group = [0,1]

#     [ Dict, [tstart, tend], coeffs, patch index ]
tc = [['activity_Phi_524_0.4.npy', [565, 580], [0,1], 1],
     ]


run='tc'
if run == 'tc':
    tc = tc[0]
    a = Analysis(tc[0])
    a.temporal_correlation(tc[2], tc[1], tc[3])
else:
    a = Analysis(data)
    #a.over_time(group, patch_index)
    #a.spatial_correlation(group, patch_index)
    #a.train_dynamics()

