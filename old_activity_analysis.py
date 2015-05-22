import matplotlib
matplotlib.rcParams.update({'figure.autolayout': True}) # Magical tight layout

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

    def __init__(self, phi, image_data_name):
        parr = phi.split('_')
        direc = parr[0] + '_' + parr[1]
        self.phi_name = phi
        self.phi = scipy.io.loadmat('dict/%s/%s' % (direc,phi))['Phi']
        self.sz = np.sqrt(self.phi.shape[0])
        self.neurons = self.phi.shape[1]

        self.image_data_name = image_data_name
        self.images = scipy.io.loadmat('mat/%s.mat' % image_data_name)[image_data_name]
        (self.imsize, imsize, self.num_images) = np.shape(self.images)
        self.patch_per_dim = int(np.floor(imsize / self.sz))

        self.activity_log = np.load('activity/activity_%s_%s.npy' % (phi, image_data_name))
        self.membrane_log = np.load('activity/membrane_%s_%s.npy' % (phi, image_data_name))
        self.nframes = self.activity_log.shape[2] # num_frames
        self.patches = self.activity_log.shape[1] # batch_size
        self.lambdav = 0.2

        plt.ion()

    def thresh(self, u):
        'LCA threshold function'
        #theta = thnp.ones(self.patches) * self.lambdav
        theta = self.lambdav
        a = abs(u) - theta;
        a[a < 0] = 0
        a = np.sign(u) * a
        return a

    def find_active_pi(self, coeffs, patch_i, tstart, tend):
        best_coeffs = -1
        best_pi = -1
        best_activity = -1

        group_size = 1
        if coeffs is not None:
            all_coeffs = [coeffs]
        else:
            all_coeffs = [range(i, i+group_size) for i in range(0, self.neurons, group_size)]
        all_patches = [patch_i] if patch_i != -1 else range(self.patches)

        for coeffs in all_coeffs:
            for pi in all_patches:
                x = np.copy(np.abs(self.activity_log[coeffs, pi, tstart:tend]))
                x[x >= 1e-5] = 1
                total = np.sum(x)
                if total > best_activity:
                    best_coeffs = coeffs
                    best_pi = pi
                    best_activity = total
        print 'Best coeffs %s PI was %d' % (best_coeffs, best_pi)
        return best_coeffs, best_pi

    def find_coeff(self, coeffs, patch_i):
        for (tstart,tend) in ((0,5), (5, 10), (10, 15)):
            rows = 5
            cols = tend - tstart
            for t in range(tstart, tend):
                activities = np.copy(np.abs(self.activity_log[:, patch_i, t]))
                act_coeffs = np.fliplr([activities.argsort()])[0]
                i = 0
                for coeff in act_coeffs:
                    act = self.activity_log[coeff, patch_i, t]
                    if act == 0.0 or i >= rows:
                        break
                    img = act * self.phi[:,coeff]
                    ax = plt.subplot2grid((rows,cols), (i,t-tstart))
                    ax.set_title("T=%d, A%d" % (t,coeff))
                    ax.imshow(np.reshape(img, (self.sz, self.sz)), cmap = cm.binary, interpolation='nearest')
                    plt.savefig('most_active/%s_%d_to_%d.png' % (self.phi_name, tstart, tend))
                    #plt.draw()
                    #plt.show()
                    i = i + 1
            #plt.show(block=True)

    def power_over_time(self, patch_i):
        tstart = 0
        tend   = 100

        for t in range(tstart, tend):
            print 't=%d) Norm=%f' % (t, np.linalg.norm(self.activity_log[:, patch_i, t]))

    def over_time(self, coeffs, patch_i, tstart, tend, Z_file=None, time_only=False):
        #log = self.membrane_log
        log = self.activity_log

        if coeffs is None or patch_i == -1:
            coeffs, patch_i = self.find_active_pi(coeffs, patch_i, tstart, tend)

        if Z_file is not None:
            #Z = scipy.io.loadmat('activity/mm_%s' % Z_file)['Z']
            Z = scipy.io.loadmat('activity/aa_%s' % Z_file)['Z']
            #Z = Z[coeffs][:,coeffs]
            #pdb.set_trace()
            #Z = scipy.io.loadmat('dict/Phi_568/Phi_568_1.6.mat')['Z']
            #Z = scipy.io.loadmat('dict/Phi_573/Phi_573_1.0.mat')['Z']
            #Z = scipy.io.loadmat('dict/Phi_575/Phi_575_1.0.mat')['Z']
            #Z = scipy.io.loadmat('dict/Phi_590/Phi_590_0.4.mat')['Z']
            #Z = np.eye(Z.shape[0])

            act = log[coeffs, patch_i, tstart]

            reset_after = 10
            for t in range(tstart+1, tend):
                act = np.dot(Z, act)
                #save_act = self.activity_log[coeffs, patch_i, t]
                #self.activity_log[coeffs, patch_i, t] = act
                save_act = log[coeffs, patch_i, t]
                log[coeffs, patch_i, t] = act
                #if t % reset_after == 0:
                    #act = save_act

        # Setup
        rows = 1 if time_only else 4
        cols = 4

        # Time graph
        ax_time = plt.subplot2grid((rows,cols), (0,0), colspan=cols)
        for i in coeffs:
            ax_time.plot(range(tstart, tend), [0] * (tend-tstart), color='k')
            ax_time.plot([0, 0], [-1.0, 1.0], color='k')

            ax_time.plot(range(tstart, tend), log[i, patch_i, tstart:tend], label='P_%d_A%d' % (patch_i, i))
            #ax_time.legend()
            #lg = ax_time.legend(bbox_to_anchor=(-0.6 , 0.40), loc=2, fontsize=10)
            #lg = ax_time.legend(bbox_to_anchor=(1.05, 1), loc=2, fontsize=50)
            #ax_time.legend(bbox_to_anchor=(0., -1.02, 1., .102), loc=3,
                       #ncol=2, fontsize=7, borderaxespad=0.)
            #lg.draw_frame(False)
            ax_time.set_title("Graphs for %s" % self.phi_name)

        if not time_only:
            # Coefficients
            for i in range(len(coeffs)):
                ax = plt.subplot2grid((rows,cols), (1,i))
                ax.imshow(np.reshape(self.phi[:,coeffs[i]], (self.sz, self.sz)), cmap = cm.binary, interpolation='nearest')
                ax.set_title("A%d" % coeffs[i])

            # Reconstruction and Images
            ax_r = plt.subplot2grid((rows,cols), (2,1)) # Reconstruct
            ax_i = plt.subplot2grid((rows,cols), (2,0)) # Images
            for iters in range(1000):
                for t in range(tstart, tend):
                    img = np.zeros(self.sz ** 2)
                    tlog = self.thresh(log[coeffs, patch_i, t])
                    img = np.dot(self.phi[:, coeffs], tlog)

                    ax_r.set_title("Partial Reconstruction t=%d" % t)
                    ax_r.imshow(np.reshape(img, (self.sz, self.sz)), cmap = cm.binary, interpolation='nearest')


                    ax_i.set_title("Image t=%d" % t)
                    r = np.floor(patch_i / self.patch_per_dim)
                    c = patch_i % self.patch_per_dim
                    #r = 11
                    #c = 13
                    rr = r * self.sz
                    cc = c * self.sz
                    ax_i.imshow(self.images[rr:rr+self.sz, cc:cc+self.sz, t].T, cmap = cm.binary, interpolation='nearest')
                    plt.draw()
                    #plt.show(block=True)

        plt.show(block=True)

    def spatial_correlation(self, coeffs, patch_i, log=False):
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

    def get_eta(self, t):
        batch_size = 1
        start = 600
        inc = 600
        start_val = 0.400

        for i in range(10):
            if t < start + i * inc:
                return start_val/batch_size
            start_val /= 2.0
        return start_val/batch_size

    def train_dynamics(self, coeffs, patch_i, tstart, tend):
        if coeffs == None:
            coeffs = range(self.neurons)
        neurons = len(coeffs)

        name = 'aa'
        data = self.activity_log[coeffs, patch_i, tstart:tend]
        #name = 'mm'
        #data = self.membrane_log[coeffs, patch_i, tstart:tend]

        Z = np.zeros((neurons, neurons))
        name = 'activity/%s_%s_%s_%d_to_%d' % (name, self.phi_name, self.image_data_name, tstart, tend)

        # Obj a - Za
        for t in range(3000):
            print 'T=%d' % t
            R_sum = np.zeros(len(data[:,0]))
            for i in range(1, tend - tstart):
                a_prev, a = (data[:, i-1], data[:, i])
                R = a - np.dot(Z, a_prev)
                #print '\ti=%d, R1=%f' % (i, np.sqrt(np.sum(np.abs(R))))
                Z = Z + self.get_eta(t) * np.dot(R.reshape(neurons,1) , a_prev.reshape(1,neurons))
                R_sum += R
                #R2 = a - np.dot(Z, a_prev)
                #print '\ti=%d, R2=%f' % (i, np.sqrt(np.sum(np.abs(R2))))
                #print ''

            print '\ti=%d, R_sum=%f' % (i, np.sqrt(np.sum(np.abs(R_sum))))
            #if t > 0 and t % 200 == 0:
                #scipy.io.savemat(name, {'Z': Z})

        scipy.io.savemat(name, {'Z': Z})

        #pdb.set_trace()
        print 'Z', Z

    def temporal_correlation(self, coeffs, time, patch_i):
        if len(coeffs) != 2:
            raise Exception("Greater than 2 dim not supported.")

        tstart, tend = time[0], time[1]
        pi = patch_i
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

def get_neighbors(coeff, dist=2):
    group = []

    for col in range(-dist, dist):
        for row in range(-dist, dist):
            group.append(coeff + col * 18 + row * 1)
    return group

#data  = 'Phi_520_0.6'
#group = [0,1,2,3]

#data  = 'Phi_524_0.4'
#grou = [0,1]
#group = [46,47] # Strong correlatioN
#group = [0,1]

#    [  Dict,         [tstart, tend], coeffs, patch index ]
#tc = [['Phi_524_0.4', [565, 580], [0,1], 1],
     #]


#phi   = 'Phi_525_0.5'
#group = [0,1,2]

#patch_i = 189
#phi   = 'Phi_520_0.6'
#group = [303, 311, 203, 574, 339, 337, 575, 481, 550, 272, 435, 433, 434]

#patch_i = 189
#phi   = 'Phi_524_0.4'
#group = [385, 384, 509, 508, 489, 141, 508, 460, 252, 104, 118, 5, 567, 260, 204, 205]
##group = [385, 384, 509, 508, 489, 141, 508, 460, 252, 104, 118, 5, 567, 260, 204, 205]
#group = [385, 508, 489, 141, 460, 252, 104, 5, 567, 204, 205]

#image_data_name = 'IMAGES_DUCK_SHORT'
#image_data_name = 'IMAGES_DUCK'

image_data_name = 'IMAGES_EDGE_DUCK_2'
patch_i = 189
phi    = 'Phi_463_0.3'
tstart = 0
tend   = 100
z_file = '%s_%s_%d_to_%d' % (phi, image_data_name, tstart, tend)
#group = get_neighbors(218)
#group = [219, 199, 181, 301, 284, 38, 180, 0, 18, 120, 101, 58, 59] # Interesting patterns
group = None
#group = [219, 181, 18, 180, 0, 120]
#group = [180, 0, 120]
#group = [180, 120]
#group = [0, 180]

a = Analysis(phi, image_data_name)
#a.power_over_time(patch_i)
#a.over_time(range(a.neurons), patch_i, tstart, tend)
a.over_time(range(a.neurons), patch_i, tstart, tend, z_file)
#a.over_time(group, patch_i, tstart, tend)
#a.find_coeff(group, patch_i)
#a.spatial_correlation(group, patch_i)
#a.train_dynamics(group, patch_i, tstart, tend)
