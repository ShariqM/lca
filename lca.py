" Run LCA on a data set g"
import matplotlib
import socket
if socket.gethostname() == 'redwood2':
    matplotlib.use('Agg') # Don't crash because $Display is not set correctly on the cluster

import scipy.io
import sys
import pdb
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import cm
import time
from math import log
from math import isnan
from colors import COLORS
from datetime import datetime
from helpers import *
from theano import *
import theano.tensor as T
import h5py


# Parameters
#patch_dim   = 144 # patch_dim=(sz)^2 where the basis and patches are SZxSZ
#neurons     = 288  # Number of basis functions
patch_dim   = 256 # patch_dim=(sz)^2 where the basis and patches are SZxSZ
neurons     = 1024  # Number of basis functions
lambdav     = 0.05 # Minimum Threshold
num_trials  = 40000
batch_size  = 40
border      = 4
sz     = np.sqrt(patch_dim)

# More Parameters
runtype            = RunType.vLearning# Learning, vLearning, vReconstruct
coeff_visualizer   = False # Visualize potentials of neurons
random_patch_index = 8  # For coeff visualizer we watch a single patch over time
thresh_type        = 'hard'
coeff_eta          = 0.05
lambda_type        = ''

#image_data_name    = 'IMAGES_DUCK_LONG_SMOOTH_0.7'
image_data_name    = 'IMAGES_DUCK'
#image_data_name    = 'IMAGES_DUCK_SMOOTH_0.7'
#image_data_name    = 'IMAGES'
iters_per_frame    = 10 # Only for vLearning
if image_data_name == 'IMAGES_DUCK_LONG':
    # Load data, have to use h5py because had to use v7.3 because .mat is so big.
    f = h5py.File('mat/%s.mat' % image_data_name, 'r',)
    IMAGES = np.array(f.get(image_data_name))
    IMAGES = np.swapaxes(IMAGES, 0, 2) # v7.3 reorders for some reason, or h5?
else:
    IMAGES = scipy.io.loadmat('mat/%s.mat' % image_data_name)[image_data_name]
(imsize, imsize, num_images) = np.shape(IMAGES)

#init_Phi = ''
#init_Phi = 'Phi_32/Phi_32_22.5.mat'
#init_Phi = 'Phi_52/Phi_52_1.2.mat'
#init_Phi = 'Phi_54/Phi_54_1.0.mat'
#init_Phi = 'Phi_67/Phi_67_1.2.mat'
#init_Phi = 'Phi_.mat/Phi_11.mat'
init_Phi = 'Phi_.mat/Phi_6/Phi_67/Phi_67_1.2.mat'
init_Phi = 'Phi_71/Phi_71_7.5'
#init_Phi = 'Phi_IMAGES_DUCK_OC=4.0_lambda=0.007.mat' # Solid dictionary

load_sequentially = False # False unsupported at the moment 2-6-15
time_batch_size = 200

print 'num images %d, num trials %d' % (num_images, num_trials)

if coeff_visualizer:
    print 'Setting batch size to 1'
    batch_size = 1

# Don't block when showing images
plt.ion()

# Load video images from sequential time points
def load_vImages(I, t, patch_per_dim):
    if coeff_visualizer: # Pick 1 specified patch
        print 'coeff visualizer'
        rr = random_patch_index * sz
        cc = random_patch_index * sz
        I[:,0] = np.reshape(IMAGES[rr:rr+sz, cc:cc+sz, t], patch_dim, 1)
    else: # Grab all the patches
        i = 0
        for r in range(patch_per_dim):
            for c in range(patch_per_dim):
                rr = r * sz
                cc = c * sz
                I[:,i] = np.reshape(IMAGES[rr:rr+sz, cc:cc+sz, t], patch_dim, 1)
                i = i + 1
    return I

# Load video images from random starting points
def load_vrImages(I):
    # Choose a random image less than batch_size images away from the end
    imi = np.ceil((num_images - batch_size) * random.uniform(0, 1))

    if coeff_visualizer: # Pick 1 specified patch
        print 'coeff visualizer'
        rr = random_patch_index * sz
        cc = random_patch_index * sz
        I[:,0] = np.reshape(IMAGES[rr:rr+sz, cc:cc+sz, imi], patch_dim, 1)
    else:
        r = border + np.ceil((imsize-sz-2*border) * random.uniform(0, 1))
        c = border + np.ceil((imsize-sz-2*border) * random.uniform(0, 1))
        for i in range(batch_size):
            pdb.set_trace()
            I[:,i] = np.reshape(IMAGES[r:r+sz, c:c+sz, imi-1+i], patch_dim, 1)
            i = i + 1
    return I

def load_images(I):
    # Choose a random image
    imi = np.ceil(num_images * random.uniform(0, 1))

    # Pick batch_size random patches from the random image
    for i in range(batch_size):
        r = border + np.ceil((imsize-sz-2*border) * random.uniform(0, 1))
        c = border + np.ceil((imsize-sz-2*border) * random.uniform(0, 1))

        I[:,i] = np.reshape(IMAGES[r:r+sz, c:c+sz, imi-1], patch_dim, 1)

    return I

import os
def log_and_save_dict(Phi, comp):
    # Log dictionary and Save Mat file
    f = open('log.txt', 'r')
    rr = 0
    while True:
        r = f.readline()
        if r == '':
            break
        rr = r
    f.close()

    if comp == 0.0: # Only log on first write
        name = 'Phi_%d' % (int(rr) + 1)

        path = 'dict/%s' % name
        if not os.path.exists(path):
            os.makedirs(path)

        f = open('log.txt', 'a')

        f.write('\n*** %s ***\n' % name)
        f.write('Time=%s\n' % datetime.now())
        f.write('RunType=%s\n' % get_RunType_name(runtype))
        f.write('IMAGES=%s\n' % image_data_name)
        f.write('patch_dim=%d\n' % patch_dim)
        f.write('neurons=%d\n' % neurons)
        f.write('lambdav=%.3f\n' % lambdav)
        f.write('num_trials=%d\n' % num_trials)
        f.write('batch_size=%d\n' % batch_size)
        f.write('time_batch_size=%d\n' % time_batch_size)
        f.write('NUM_IMAGES=%d\n' % num_images)
        f.write('iter_per_frame=%d\n' % iters_per_frame)
        f.write('thresh_type=%s\n' % thresh_type)
        f.write('coeff_eta=%.3f\n' % coeff_eta)
        f.write('lambda_type=[%s]\n' % lambda_type)
        f.write('InitPhi=%s\n' % init_Phi)
        f.write('Load Sequentially=%s\n' % load_sequentially)

        f.write('%d\n' % (int(rr)+1))
        f.close()

        logfile = '%s/%s_log.txt' % (path, name)
        print 'Assigning stdout to %s' % logfile
        sys.stdout = open(logfile, 'w')
    else:
        name = 'Phi_%d' % (int(rr))
        path = 'dict/%s' % name

    scipy.io.savemat('%s/%s_%.1f' % (path, name, comp), {'Phi':Phi})
    print '%s_%.1f successfully written.' % (name, comp)

from showbfs import showbfs
def Learning():
    global batch_size # Wow epic fail http://bugs.python.org/issue9049
    global lambdav
    global num_images

    # Initialize basis functions
    if init_Phi != '':
        Phi = scipy.io.loadmat('dict/%s' % init_Phi)
        Phi = Phi['Phi']
    else:
        Phi = np.random.randn(patch_dim, neurons)
        Phi = np.dot(Phi, np.diag(1/np.sqrt(np.sum(Phi**2, axis = 0))))

    ahat_prev = None # For reusing coefficients of the last frame
    if runtype == RunType.vLearning:
        patch_per_dim = int(np.floor(imsize / sz))
        if not coeff_visualizer and load_sequentially:
            batch_size = patch_per_dim**2

    # Initialize batch of images
    I = np.zeros((patch_dim, batch_size))

    max_active = float(neurons * batch_size)
    start = datetime.now()

    if runtype == RunType.Learning:
        for tt in range(num_trials):
            if runtype == RunType.Learning:
                t = tt
                I = load_images(I)
                ahat = sparsify(I, Phi, lambdav) # Coefficient Inference
            else:
                t = tt % num_images
                if load_sequentially:
                    I = load_vImages(I, t, patch_per_dim)
                else:
                    I = load_vrImages(I)

                ahat = sparsify(I, Phi, lambdav, ahat_prev=ahat_prev,
                                   num_iterations=iters_per_frame)

            #if tt == 400:
                #log_and_save_dict(Phi, 100.0 * float(tt)/num_trials)
                #pdb.set_trace()

            # Calculate Residual
            R = I - np.dot(Phi, ahat)

            # Update Basis Functions
            dPhi = get_eta(tt, runtype, batch_size) * (np.dot(R, ahat.T))

            Phi = Phi + dPhi
            Phi = np.dot(Phi, np.diag(1/np.sqrt(np.sum(Phi**2, axis = 0))))

            # Plot every 200 iterations
            # Plot every 1   iterations
            #if tt > 400 or np.mod(tt, 200) == 0:
            if np.mod(tt, 200) == 0:
                try:
                    var = I.var().mean()
                    mse = (R ** 2).mean()
                    snr = 10 * log(var/mse, 10)

                    if isnan(snr):
                        pdb.set_trace()

                    ahat_c = np.copy(ahat)
                    ahat_c[np.abs(ahat_c) > lambdav/1000.0] = 1
                    ac = np.sum(ahat_c)

                    print '%.4d) lambdav=%.3f || snr=%.2fdB || AC=%.2f%% || ELAP=%d' \
                            % (tt, lambdav, snr, 100.0 * ac / max_active,
                               (datetime.now() - start).seconds)
                except Exception as e:
                    pdb.set_trace()

                sys.stdout.flush()
                showbfs(Phi)
                plt.show()

            if np.mod(tt, 1000) == 0:
                log_and_save_dict(Phi, 100.0 * float(tt)/num_trials)

            ahat_prev = ahat
    else:
        for tt in range(num_trials):

            VI = np.zeros((patch_dim, batch_size, time_batch_size)) # Batch_size videos of time_batch_size frames
            for x in range(batch_size):
                # Choose a random image less than time_batch_size images away from the end
                imi = np.floor((num_images - time_batch_size) * random.uniform(0, 1))
                r = border + np.ceil((imsize-sz-2*border) * random.uniform(0, 1))
                c = border + np.ceil((imsize-sz-2*border) * random.uniform(0, 1))
                VI[:,x,:] = np.reshape(IMAGES[r:r+sz, c:c+sz, imi:imi+time_batch_size], (patch_dim, time_batch_size), 1)
                #showbfs(VI[:,x,:])
                #pdb.set_trace()

            ahat_prev = None
            for i in range(time_batch_size):
                I = VI[:,:,i]

                ahat = sparsify(I, Phi, lambdav, ahat_prev=ahat_prev,
                                num_iterations=iters_per_frame)

                # Calculate Residual
                R = I - np.dot(Phi, ahat)

                # Update Basis Functions
                dPhi = get_eta(batch_size * tt, runtype, time_batch_size) * (np.dot(R, ahat.T))

                Phi = Phi + dPhi
                Phi = np.dot(Phi, np.diag(1/np.sqrt(np.sum(Phi**2, axis = 0))))

                ahat_c = np.copy(ahat)
                ahat_c[np.abs(ahat_c) > lambdav/1000.0] = 1
                ac = np.sum(ahat_c)

                ahat_prev = ahat

                if i % 50 == 0:
                    print '\t%.3d) lambdav=%.3f || AC=%.2f%%' % (i, lambdav, 100.0 * ac / max_active)


            var = I.var().mean()
            mse = (R ** 2).mean()
            snr = 10 * log(var/mse, 10)

            sys.stdout.flush()
            showbfs(Phi)
            plt.show()

            if np.mod(tt, 5) == 0:
                log_and_save_dict(Phi, 100.0 * float(tt)/num_trials)

            print '%.4d) lambdav=%.3f || snr=%.2fdB || AC=%.2f%% || ELAP=%d' \
                        % (tt, lambdav, snr, 100.0 * ac / max_active,
                           (datetime.now() - start).seconds)

    log_and_save_dict(Phi, 100.0)
    plt.show()

def vReconstruct():
    global batch_size

    # Just look at first X frames
    num_frames = 60

    # Load dict from Learning run
    Phi = scipy.io.loadmat('dict/%s' % init_Phi)['Phi']

    # Tile the entire image with patches
    patch_per_dim = int(np.floor(imsize / sz))

    # Set the batch_size to number of patches in an image
    if not coeff_visualizer:
        batch_size = patch_per_dim**2

    # Initialize batch of images
    I = np.zeros((patch_dim, batch_size))

    # **** Hack? Diff lambda for training vs. reconstructing ****
    global lambdav
    lambdav = 0.02

    max_active = float(neurons * batch_size)

    # Run parameters: (bool=Initialize coeff to prev frame, int=iters_per_frame, float=lambdav)
    #run_p = [(True, 10, 0.02), (True, 20, 0.02), (True, 40, 0.02)]
    run_p = [(False, 80, lambdav)]
    #run_p = [(True, 120), (True, 90), (True, 60)]
    #run_p = [(True, 120), (True, 40), (True, 20), (False, 120), (False, 40), (False, 20)]
    #run_p = [(True, 120), (True, 40), (True, 20)]
    #run_p = [(True, 60)]
    #run_p = [(True, 10), (True, 30), (False, 60)]
    #run_p = [(False, 80), (True, 5), (True, 3), (True, 1)]

    labels = []
    for (initP, x, y) in run_p:
        if initP == True:
            labels += ["InitP (%d)" % x]
        else:
            labels += ["Init0 (%d)" % x]

    runs = len(labels)
    rcolor = [COLORS['red'], COLORS['green'], COLORS['blue'], COLORS['black'],
              COLORS['yellow2'], COLORS['purple']]
    top_CC_AC=0 # [C]hanged [C]oefficients, [A]ctive [C]oefficients

    for run in range(runs):
        # Record data
        MSE = [] # Mean Squared Error
        SNR = [] # Signal to Noise ratio
        AC  = np.zeros(num_frames) # Active coefficients
        CC  = np.zeros(num_frames) # Changing coefficients

        ahat_prev = np.zeros((neurons, batch_size))
        ahat_prev_c = np.zeros((neurons, batch_size))

        #lambdav = run_p[run][2]

        start = datetime.now()
        for t in range(num_frames):
            I = load_vImages(I, t, patch_per_dim)

            if run_p[run][0] == True: # InitP
                ahat = sparsify(I, Phi, lambdav, ahat_prev=ahat_prev, num_iterations=run_p[run][1])
            else:
                ahat = sparsify(I, Phi, lambdav, num_iterations=run_p[run][1])

            # Calculate Residual Error
            R = I - np.dot(Phi, ahat)
            mse = (R ** 2).mean()
            MSE.append(mse)
            var = I.var().mean()
            SNR.append(10 * log(var/mse, 10))

            ahat_prev = ahat

            ahat_c = np.copy(ahat)
            ahat_c[np.abs(ahat_c) > lambdav/1000.0] = 1
            AC[t] = np.sum(ahat_c)
            CC[t] = np.sum(ahat_c!=ahat_prev_c)
            ahat_prev_c = ahat_c

            print '%.3d) %s || lambdav=%.3f || snr=%.2fdB || AC=%.2f%%' \
                    % (t, labels[run], lambdav, SNR[t], 100.0 * AC[t] / max_active)
        elapsed = (datetime.now() - start).seconds

        plt.subplot(231)
        plt.xlabel('Time (steps)', fontdict={'fontsize':12})
        plt.ylabel('MSE', fontdict={'fontsize':12})
        plt.axis([0, num_frames, 0.0, max(MSE) * 1.1])
        plt.plot(range(num_frames), MSE, color=rcolor[run], label=labels[run])
        lg = plt.legend(bbox_to_anchor=(-0.6 , 0.40), loc=2, fontsize=10)
        lg.draw_frame(False)

        top_CC_AC = max(max(CC),max(AC),top_CC_AC) * 1.1
        plt.subplot(232)
        plt.xlabel('Time (steps)', fontdict={'fontsize':12})
        plt.ylabel('# Active Coeff', fontdict={'fontsize':12})
        plt.axis([0, num_frames, 0, top_CC_AC])
        plt.plot(range(num_frames), AC, color=rcolor[run])

        plt.subplot(233)
        plt.xlabel('Time (steps)', fontdict={'fontsize':12})
        plt.ylabel('# Changed Coeff', fontdict={'fontsize':12})
        plt.axis([0, num_frames, 0, top_CC_AC])
        plt.plot(range(num_frames), CC, color=rcolor[run])

        plt.subplot(234)
        plt.xlabel('Time (steps)', fontdict={'fontsize':12})
        plt.ylabel('SNR (dB)', fontdict={'fontsize':12})
        plt.axis([0, num_frames, 0.0, 22])
        plt.plot(range(num_frames), SNR, color=rcolor[run], label=labels[run])
        lg = plt.legend(bbox_to_anchor=(-0.6 , 0.60), loc=2, fontsize=10)
        lg.draw_frame(False)

        # % plots
        top_p = 100 * top_CC_AC / max_active
        plt.subplot(235)
        plt.xlabel('Time (steps)', fontdict={'fontsize':12})
        plt.ylabel('% Active Coeff', fontdict={'fontsize':12})
        plt.axis([0, num_frames, 0, top_p])
        plt.plot(range(num_frames), 100 * AC / max_active, color=rcolor[run])

        plt.subplot(236)
        plt.xlabel('Time (steps)', fontdict={'fontsize':12})
        plt.ylabel('% Changed Coeff', fontdict={'fontsize':12})
        plt.axis([0, num_frames, 0, top_p])
        plt.plot(range(num_frames), 100 * CC / max_active, color=rcolor[run])

    plt.suptitle("DATA=%s, LAMBDAV=%.3f, IMG=%dx%d, PAT=%dx%d, DICT=%d, PAT/IMG=%d ELAP=%d" %
                    (image_data_name, lambdav, imsize, imsize, sz, sz, neurons, patch_per_dim ** 2, elapsed), fontsize=18)
    plt.show(block=True)

def sparsenet():
    if runtype == RunType.vReconstruct:
        vReconstruct()
    else:
        Learning()

# Theano Matrix Multiplication Optimization
Gv = T.fmatrix('G')
av = T.fmatrix('a')
o  = T.dot(Gv, av)
f = theano.function([Gv, av], o, allow_input_downcast=True)

def visualizer_code():
    pass

def sparsify(I, Phi, lambdav, ahat_prev=None, num_iterations=80):
    """
    LCA Inference.
    I: Image batch (dim x batch)
    Phi: Dictionary (dim x dictionary element)
    lambdav: Sparsity coefficient
    coeff_eta: Update rate
    """
    global lambda_type
    batch_size = np.shape(I)[1]

    (N, M) = np.shape(Phi)
    sz = np.sqrt(N)

    b = f(Phi.T, I)
    G = f(Phi.T, Phi) - np.eye(M)

    if runtype == RunType.Learning:
        lambda_type = 'l = 0.5 * np.max(np.abs(b), axis = 0)'
        l = 0.5 * np.max(np.abs(b), axis = 0)
    else:
        #l = 0.1 * np.max(np.abs(b), axis = 0)
        lambda_type = 'Fixed and lambdav'
        l = np.ones(batch_size)
        l *= lambdav

    if ahat_prev is not None:
        u = ahat_prev
        #u = np.dot(ahat_prev, np.diag(l)) # Artificially set at threshold
    else:
        u = np.zeros((M,batch_size))

    a = g(u, l)

    if coeff_visualizer:
        assert batch_size == 1
        plt.subplot(313)
        coeffs = plt.bar(range(M), u, color='b')
        lthresh = plt.plot(range(M), list(l) * M, color='r')

        if runtype == RunType.Learning:
            plt.axis([0, M, 0, 1.05])
        else:
            plt.axis([0, M, 0, lambdav * 10])

        plt.subplot(312)
        plt.imshow(np.reshape(I[:,0], (sz, sz)),cmap = cm.binary, interpolation='nearest')

        plt.subplot(311)
        recon = np.dot(Phi, a)
        plt.imshow(np.reshape(recon, (sz, sz)),cmap = cm.binary, interpolation='nearest')

        plt.draw()
        plt.show()
        time.sleep(1)

    for t in range(num_iterations):
        u = coeff_eta * (b - f(G,a)) + (1 - coeff_eta) * u
        a = g(u, l)

        l = 0.95 * l
        l[l < lambdav] = lambdav

        if coeff_visualizer:
            for coeff, i in zip(coeffs, range(M)):
                coeff.set_height(u[i])
            lthresh[0].set_data(range(M), list(l) * M)

            plt.subplot(311)
            recon = np.dot(Phi, a)
            plt.imshow(np.reshape(recon, (sz, sz)),cmap = cm.binary, interpolation='nearest')

            plt.title('Iter=%d/%d' % (t, num_iterations))
            plt.draw()
            plt.show()
# Do some clf magic

    if coeff_visualizer:
        plt.close()

    return a

def g(u,theta):
    """
    LCA threshold function
    u: coefficients
    theta: threshold value
    """
    if thresh_type=='hard': # L0 Approximation
        a = u;
        a[np.abs(a) < theta] = 0
    elif thresh_type=='soft': # L1 Approximation
        a = abs(u) - theta;
        a[a < 0] = 0
        a = np.sign(u) * a
    return a

sparsenet()
