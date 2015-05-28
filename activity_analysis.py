import matplotlib
matplotlib.rcParams.update({'figure.autolayout': True}) # Magical tight layout

import math
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

#class DataSet():
    #def __init__(self, log, tstart, tend):
        #self.log    = log
        #self.tstart = tstart
        #self.tend   = tend

dtype = theano.config.floatX
class Analysis():

    LOG_NAME = 'aalog.txt'
    datasets = [
                #'IMAGES_DUCK'
                ['IMAGES_DUCK_SHORT', 0, 100],#
                #['IMAGES_EDGE_DUCK',           0, 100],
                #['IMAGES_EDGE_DUCK', 0, 100],
                #['IMAGES_EDGE_DUCK_r=20_c=20', 0, 100],
                #['IMAGES_EDGE_DUCK_r=21_c=21', 0, 100],
                #['IMAGES_EDGE_DUCK_r=22_c=22', 0, 100],
                #['IMAGES_EDGE_DUCK_r=23_c=23', 0, 100],
                #['IMAGES_EDGE_DUCK_r=24_c=24', 0, 100],
               ]
    patches = 2
    reconstruct_i = 2 # Which dataset index to reconstruct (in over_time())
    reset_after = 15
    cells = 2 # Number of Gamma's
    clambdav = 0.00
    citers  = 20
    normalize_Gamma = True
    #reset_after = -1 # -1 means never
    inertia = True

    a_mode  = True # ahat or u mode
    patch_i   = 189
    phi_name  = 'Phi_463_0.3'
    coeffs    = None

    # Training parameters
    num_trials = 3000
    eta_init   = 0.200
    eta_inc    = 1200

    ceta       = 0.05
    log_often  = 8
    Z_init     = ''
    G_init     = ''
    #G_init     = 'AA_LOG_80'
    #Z_init     = 'AA_Log_7'

    coeff_visualizer = False
    graphics_initialized = False

    def __init__(self):
        parr = self.phi_name.split('_')
        direc = parr[0] + '_' + parr[1]
        self.phi = scipy.io.loadmat('dict/%s/%s' % (direc,self.phi_name))['Phi']

        self.sz = np.sqrt(self.phi.shape[0])
        self.neurons = self.phi.shape[1]
        #self.neurons = 100
        if self.coeffs == None:
            self.coeffs = range(self.neurons)

        if self.coeff_visualizer:
            self.patches = 1

        'Log of ahat or u values'
        #self.logs = []
        #self.images = []
        aname = 'activity' if self.a_mode else 'membrane'
        for ds in self.datasets:
            #idn = ds[0]
            #self.images.append(scipy.io.loadmat('mat/%s.mat' % idn)[idn])
            #(self.imsize, imsize, self.num_images) = np.shape(self.images[0])
            #self.patch_per_dim = int(np.floor(imsize / self.sz))

            self.log = np.load('activity/%s_%s_%s.npy' % (aname, self.phi_name, ds[0]))
            self.log = self.log[self.coeffs, 0:self.patches, ds[1]:ds[2]]
            #self.logs.append(log)

        self.timepoints = self.log.shape[2] # num_frames FIXME
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
        plt.ion()
        self.init_theano()

    def init_theano(self):
        if True:
            Ahat = T.fmatrix('Ahat')
            A_prev = T.fmatrix('A_prev')
            Gam = T.tensor3('Gam')
            C = T.fmatrix('C')
            ##E = 0.5 * ((Ahat - T.batched_dot(T.tensordot(Gam, C, 1).dimshuffle(2, 0, 1), A_prev.T)).norm(2) ** 2)
            #E = 0.5 * ((T.batched_dot(T.tensordot(Gam, C, 1).dimshuffle(2, 0, 1), A_prev.T)).norm(2) ** 2)
            E = 0.5 * ((Ahat - T.batched_dot(T.tensordot(Gam, C, 1).dimshuffle(2, 0, 1), A_prev.T).T).norm(2) ** 2)
            #E = 0.5 * (T.tensordot(Gam, C, 1).dimshuffle(2, 0, 1).norm(2) ** 2)
            gc = T.grad(E, C)
            self.gc = function([Ahat, A_prev, Gam, C], gc, allow_input_downcast=True)
            #self.gc = function([A_prev, Gam, C], gc, allow_input_downcast=True)
            #self.gc = function([Gam, C], gc, allow_input_downcast=True)

        else:
            Ahat = T.fmatrix('Ahat')

            Gamm = np.random.randn(self.neurons, self.neurons, np.cells)
            Gam = self.Gam = theano.shared(Gammm.astype(dtype))

            Cm = np.zeros((self.cells, self.batch_size))
            C = self.C = theano.shared(Cm.astype(dtype))

            A_prev = T.fmatrix('A_prev')

            E = 0.5 * ((Ahat - T.batched_dot(T.tensordot(Gam, C, 1).dimshuffle(2, 0, 1), A_prev.T)).norm(2) ** 2)
            self.fE = function([Ahat, A_prev], E, allow_input_downcast=True)
            params = [Gam, C]
            gparams = T.grad(E, wrt=params)
            updates = adadelta_update(params, gparams)

            #self.learn_D = theano.function(inputs = [I, l],
                                    #outputs = E,
                                    #updates = [[D, updates[D]]],
                                    #allow_input_downcast=True)
#
            #self.learn_A = theano.function(inputs = [I, l],
                                    #outputs = E,
                                    #updates = [[A, updates[A]]],
                                    #allow_input_downcast=True) # Brian doesn't seem to use this





    def thresh(self, u, theta):
        'LCA threshold function'
        a = abs(u) - theta;
        a[a < 0] = 0
        a = np.sign(u) * a
        return a

    def find_active_pi(self, coeffs):
        'Find the most active coefficients and patch and return both'
        best_coeffs = -1
        best_pi = -1
        best_activity = -1

        group_size = 1
        if coeffs is not None:
            all_coeffs = [coeffs]
        else:
            all_coeffs = [range(i, i+group_size) for i in range(0, self.neurons, group_size)]
        all_patches = [patch_i] if patch_i != -1 else range(self.batch_size)

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
        'Generate a figure showing the most activie coefficients for this patch'

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

    def power_over_time(self):
        'Display the norm of the coefficients over time'
        patch_i = self.patch_i
        for t in range(self.tstart, self.tend):
            print 't=%d) Norm=%f' % (t, np.linalg.norm(self.activity_log[:, patch_i, t]))

    def over_time(self, time_only=False):
        'Plot the activity over time, the video, and the reconstruction, simultaneously'

        coeffs = self.coeffs
        patch_i = self.patch_i

        if coeffs is None or patch_i == -1:
            coeffs, patch_i = self.find_active_pi(coeffs, patch_i, tstart, tend)

        # Replace activity with predictions of data if desired
        if self.Z != None:
            for log in self.logs:
                activity = log[:, 0]
                for t in range(1, log.shape[1]):
                    if t % self.reset_after == 0:
                        activity = log[:, t]
                        continue
                    activity = np.dot(self.Z, activity)
                    log[coeffs, t] = activity # overwrite actual activity

        # Plot Setup
        log = self.logs[self.reconstruct_i]
        timepoints = self.logs[0].shape[1]
        rows = 1 if time_only else 4
        cols = 4

        # Time graph
        ax_time = plt.subplot2grid((rows,cols), (0,0), colspan=cols)
        for i in coeffs:
            ax_time.plot(range(timepoints), [0] * timepoints, color='k') # X axis
            ax_time.plot([0, 0], [-1.0, 1.0], color='k')                 # Y axis

            #ax_time.plot(range(tstart, tend), log[i, patch_i, tstart:tend], label='P_%d_A%d' % (patch_i, i))
            ax_time.plot(range(timepoints), log[i, :], label='A%d' % i)
            ax_time.axis([0, timepoints, -1, 1])
            #ax_time.legend()
            #lg = ax_time.legend(bbox_to_anchor=(-0.6 , 0.40), loc=2, fontsize=10)
            #lg = ax_time.legend(bbox_to_anchor=(1.05, 1), loc=2, fontsize=50)
            #ax_time.legend(bbox_to_anchor=(0., -1.02, 1., .102), loc=3,
                       #ncol=2, fontsize=7, borderaxespad=0.)
            #lg.draw_frame(False)
            ax_time.set_title("Graphs for %s" % self.phi_name)

        if not time_only:
            # Coefficients
            if len(coeffs) < cols:
                for i in range(len(coeffs)):
                    ax = plt.subplot2grid((rows,cols), (1,i))
                    ax.imshow(np.reshape(self.phi[:,coeffs[i]], (self.sz, self.sz)), cmap = cm.binary, interpolation='nearest')
                    ax.set_title("A%d" % coeffs[i])

            # Reconstruction and Images
            for iters in range(1000): # Show forever
                for t in range(timepoints):

                    k = 0
                    for log in self.logs:
                        ax_r = plt.subplot2grid((rows,cols), (2,k)) # Reconstruct
                        tlog = self.thresh(log[:, t])
                        img = np.dot(self.phi[:, coeffs], tlog)

                        ax_r.set_title("PRecon t=%d" % t)
                        ax_r.imshow(np.reshape(img, (self.sz, self.sz)), cmap = cm.binary, interpolation='nearest')
                        k += 1

                    for k in range(len(self.logs)):
                        ax_i = plt.subplot2grid((rows,cols), (3,k)) # Images
                        ax_i.set_title("Image_%d t=%d" % (i, t))
                        rr = (np.floor(patch_i / self.patch_per_dim)) * self.sz
                        cc = (patch_i % self.patch_per_dim) * self.sz
                        dimg = self.images[k][rr:rr+self.sz, cc:cc+self.sz, t].T
                        #dimg = self.images[self.reconstruct_i][rr:rr+self.sz, cc:cc+self.sz, t].T
                        ax_i.imshow(dimg, cmap = cm.binary, interpolation='nearest')
                    plt.draw()
                    #plt.show(block=True)

        plt.show(block=True)

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
            f.write('datasets=%s\n' % self.datasets)
            f.write('patches=%d\n' % self.patches)
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

    def csparsify(self, Gam, x_prev, x, c_prev=None):
        c = c_prev if c_prev is not None else np.zeros((self.cells, self.batch_size))
        b = csparsify_grad(x, Gam, x_prev).T
        G = G_LCA(Gam, Gam, x_prev, x_prev)

        for i in range(self.citers):
            #start = datetime.now()
            gradient = (b - np.einsum('iB, BiT->TB', c, G)) # Optimize? 23 microseconds last time
            #print 'Grad:', (datetime.now() - start).microseconds
            #gradient = -self.gc(x, x_prev, Gam, c)

            #if not np.allclose(test0, gradient, atol=1e-2):
                #print 'bug found'
                #test = np.einsum('pB,pnT,nB->TB', x, Gam, x_prev) - np.einsum('iB,pri,pnT,rB,nB->TB', c, Gam, Gam, x_prev, x_prev)
                #pdb.set_trace()
            c += self.ceta * gradient
        return 0, c

    def train_G_dynamics(self):
        'Train a Z matrix to learn the dynamics of the coefficients'
        #Z = self.Z if self.Z is not None else np.zeros((self.neurons, self.neurons))
        if self.Gam is None:
            Gam = np.zeros((self.neurons, self.neurons, self.cells))
            Gam[:,:,0] = np.eye(self.neurons) + np.random.normal(0, 0.01, (self.neurons, self.neurons))
            for i in range(1, self.cells):
                Gam[:,:,i] = np.random.normal(0, 0.25, (self.neurons, self.neurons))
            Gam = self.norm_Gam(Gam)
        else:
            Gam = self.Gam
        f = open('activity/logs/AA_%d_log.txt' % self.log_idx, 'w')

        start = datetime.now()
        for k in range(self.num_trials):
            R_sum = np.zeros((self.neurons, self.batch_size))

            c_prev = None
            v_prev = np.zeros((self.cells, self.batch_size))
            for t in range(1, self.timepoints):
                # Data
                x_prev, x = (self.log[:, :, t-1], self.log[:, :, t])

                # Inference
                v, chat = self.csparsify(Gam, x_prev, x, c_prev=c_prev)

                # Residual
                x_pred = gam_predict(Gam, chat, x_prev).T
                R = x - x_pred
                #print '\t\tR=%f' % (np.sum(np.abs(R)) / self.batch_size)

                dGam = t3tendot2(R, chat, x_prev)
                #print np.max(np.abs(dGam))
                Gam += self.get_eta(k) * dGam

                Gam = self.norm_Gam(Gam)

                R_sum += np.abs(R)
                v_prev = v
                c_prev = chat if self.inertia else None

                #v, chat = 0, csparsify_grad(x, Gam, x_prev).T
                #x_pred = gam_predict(Gam, chat, x_prev).T
                #R = x - x_pred
                #print '\t\tR After=%f' % (np.sum(np.abs(R)) / self.batch_size)

            e = (datetime.now() - start).seconds
            r = np.sum(R_sum) / (self.timepoints * self.batch_size)
            msg = 'T=%.4d E=%ds, R=%f, c=%f' % (k, e, r, np.average(np.abs(chat)))
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
    # Retrieve the neighbor indices of a topographic group
    group = []

    for col in range(-dist, dist):
        for row in range(-dist, dist):
            group.append(coeff + col * 18 + row * 1)
    return group

a = Analysis()
#a.over_time()
#a.train_dynamics()
a.train_G_dynamics()
