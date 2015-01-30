" Run LCA on a data set g"
import matplotlib

redwood2_run = True
if redwood2_run:
    matplotlib.use('Agg') # Don't crash because $Display is not set correctly on the cluster
import scipy.io
import pdb
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import cm
import time
from math import log
from colors import COLORS
from datetime import datetime
from helpers import *
from theano import *
import theano.tensor as T
import h5py


# Parameters
patch_dim   = 144 # patch_dim=(sz)^2 where the basis and patches are SZxSZ
neurons     = 288  # Number of basis functions
#patch_dim   = 256 # patch_dim=(sz)^2 where the basis and patches are SZxSZ
#neurons     = 1024  # Number of basis functions
lambdav     = 0.05 # Minimum Threshold
num_trials  = 500
batch_size  = 100
border      = 4
sz     = np.sqrt(patch_dim)

# More Parameters
runtype            = RunType.learning # learning, rt_learning, rt_reconstruct
coeff_visualizer   = False # Visualize potentials of neurons
random_patch_index = 8  # For coeff visualizer we watch a single patch over time
thresh_type        = 'hard'
coeff_eta          = 0.05
lambda_type        = ''

#image_data_name    = 'IMAGES_DUCK_LONG_SMOOTH_0.7'
image_data_name    = 'IMAGES_DUCK_LONG'
#image_data_name    = 'IMAGES'
iters_per_frame    = 10 # Only for rt_learning
if image_data_name == 'IMAGES_DUCK_LONG':
    # Load data, have to use h5py because had to use v7.3 because .mat is so big.
    f = h5py.File('mat/%s.mat' % image_data_name, 'r',)
    IMAGES = np.array(f.get(image_data_name))
    IMAGES = np.swapaxes(IMAGES, 0, 2) # v7.3 reorders for some reason, or h5?
else:
    IMAGES = scipy.io.loadmat('mat/%s.mat' % image_data_name)[image_data_name]
(imsize, imsize, num_images) = np.shape(IMAGES)

print 'num images %d, num trials %d' % (num_images, num_trials)

if coeff_visualizer:
    print 'Setting batch size to 1 for coeff visualizer'
    batch_size = 1

# Don't block when showing images
plt.ion()

def load_rt_images(I, t, patch_per_dim):
    if coeff_visualizer: # Pick 1 specified patch
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

def load_images(I, t):
    # Choose a random image
    imi = np.ceil(num_images * random.uniform(0, 1))

    # Pick batch_size random patches from the random image
    for i in range(batch_size):
        r = border + np.ceil((imsize-sz-2*border) * random.uniform(0, 1))
        c = border + np.ceil((imsize-sz-2*border) * random.uniform(0, 1))

        I[:,i] = np.reshape(IMAGES[r:r+sz, c:c+sz, imi-1], patch_dim, 1)

    return I

def graph_basis(R, I, Phi, start, t):
    mse = (R ** 2).mean()
    var = I.var().mean()
    print "Iteration %.4d, Var = %.2f, SNR = %.2fdB ELAP=%d"  \
            % (t, var, 10 * log(var/mse, 10), (datetime.now() - start).seconds)

    side = np.sqrt(neurons)
    image = np.zeros((sz*side+side,sz*side+side))
    for i in range(side.astype(int)):
        for j in range(side.astype(int)):
            patch = np.reshape(Phi[:,i*side+j],(sz,sz))
            patch = patch/np.max(np.abs(patch))
            image[i*sz+i:i*sz+sz+i,j*sz+j:j*sz+sz+j] = patch

    plt.imshow(image, cmap=cm.Greys_r, interpolation="nearest")
    plt.draw()
    plt.show()

def log_and_save_dict(Phi):
    # Log dictionary and Save Mat file
    f = open('log.txt', 'r')
    rr = 0
    while True:
        r = f.readline()
        if r == '':
            break
        rr = r
    name = 'Phi_%d' % (int(rr) + 1)

    f.close()
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
    f.write('NUM_IMAGES=%d\n' % num_images)
    f.write('iter_per_frame=%d\n' % iters_per_frame)
    f.write('thresh_type=%s\n' % thresh_type)
    f.write('coeff_eta=%.3f\n' % coeff_eta)
    f.write('lambda_type=[%s]\n' % lambda_type)

    f.write('%d\n' % (int(rr)+1))
    f.close()

    scipy.io.savemat('dict/%s' % name, {'Phi':Phi})
    print '%s successfully written.' % name

def learning():
    global batch_size # Wow epic fail http://bugs.python.org/issue9049
    global lambdav
    global num_images

    # Initialize basis functions
    if False:
        Phi = scipy.io.loadmat('dict/Phi_10.mat')
        Phi = Phi['Phi']
    else:
        Phi = np.random.randn(patch_dim, neurons)
        Phi = np.dot(Phi, np.diag(1/np.sqrt(np.sum(Phi**2, axis = 0))))

    ahat_prev = None # For reusing coefficients of the last frame
    if runtype == RunType.rt_learning:
        lambdav = 0.05
        patch_per_dim = int(np.floor(imsize / sz))
        if not coeff_visualizer:
            batch_size = patch_per_dim**2

    # Initialize batch of images
    I = np.zeros((patch_dim, batch_size))

    max_active = float(neurons * batch_size)
    start = datetime.now()
    for tt in range(num_trials):
        if runtype == RunType.learning:
            t = tt
            I = load_images(I, t)
            ahat = sparsify(I, Phi, lambdav) # Coefficient Inference
        else:
            t = tt % num_images
            I = load_rt_images(I, t, patch_per_dim)
            ahat = sparsify(I, Phi, lambdav, ahat_prev=ahat_prev,
                               num_iterations=iters_per_frame)

        # Calculate Residual
        R = I - np.dot(Phi, ahat)

        # Update Basis Functions
        dPhi = get_eta(t, batch_size) * (np.dot(R, ahat.T))

        Phi = Phi + dPhi
        Phi = np.dot(Phi, np.diag(1/np.sqrt(np.sum(Phi**2, axis = 0))))

        # Plot every 400 iterations
        if np.mod(t, 400) == 0:
            graph_basis(R, I, Phi, start, t)
        if np.mod(t, 200) == 0 and runtype == RunType.rt_learning:
            var = I.var().mean()
            mse = (R ** 2).mean()
            snr = 10 * log(var/mse, 10)

            ahat_c = np.copy(ahat)
            ahat_c[np.abs(ahat_c) > lambdav/1000.0] = 1
            ac = np.sum(ahat_c)

            print '%.3d) lambdav=%.3f || snr=%.2fdB || AC=%.2f%%' \
                    % (t, lambdav, snr, 100.0 * ac / max_active)

        ahat_prev = ahat

    log_and_save_dict(Phi)
    plt.show()

def rt_reconstruct():
    global batch_size

    # Just look at first 200
    num_frames = 60

    # Load dict from learning run
    Phi = scipy.io.loadmat('dict/Phi_9.mat')
    Phi = Phi['Phi']

    # Tile the entire image with patches
    patch_per_dim = int(np.floor(imsize / sz))

    # Set the batch_size to number of patches in an image
    if not coeff_visualizer:
        batch_size = patch_per_dim**2

    # Initialize batch of images
    I = np.zeros((patch_dim, batch_size))

    # **** Hack? Diff lambda for training vs. reconstructing ****
    lambdav = 0.01

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

        lambdav = run_p[run][2]

        start = datetime.now()
        for t in range(num_frames):
            I = load_rt_images(I, t, patch_per_dim)

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
    if runtype == RunType.rt_reconstruct:
        rt_reconstruct()
    else:
        learning()

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

    if runtype == RunType.learning:
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

        if runtype == RunType.learning:
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
