import scipy.io
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import cm
import time
from math import log
from colors import COLORS
from datetime import datetime

def get_eta(t,batch_size):
    start = 1200
    if t < start:
        return 6.0/batch_size
    if t < start+500:
        return 3.0/batch_size
    if t < start+1000:
        return 1.0/batch_size
    if t < start+1500:
        return 0.5/batch_size
    if t < start+2000:
        return 0.25/batch_size
    if t < start+2500:
        return 0.10/batch_size
    return 0.10/batch_size

run_learning=False
coeff_visualizer=False
#def sparsenet(patch_dim=64, neurons=128, lambdav=0.01, num_trials=5000, batch_size=100):
def sparsenet(patch_dim=256, neurons=1024, lambdav=0.007, num_trials=5000, batch_size=100):

    """
    N: # Inputs
    M: # Outputs
    lambdav: Sparsity Constraint
    eta: Learning Rate
    num_trials: Learning Iterations
    batch_size: Batch size per iteration
    border: Border when extracting image patches
    """
    border=4
    name = 'IMAGES_DUCK'
    IMAGES = scipy.io.loadmat('mat/%s.mat' % name)[name]

    (imsize, imsize, num_images) = np.shape(IMAGES)

    if coeff_visualizer and batch_size != 1:
        print 'Setting batch size to 1 for coeff visualizer'
        batch_size = 1

    sz = np.sqrt(patch_dim)

    # Initialize basis functions
    Phi = np.random.randn(patch_dim, neurons)
    Phi = np.dot(Phi, np.diag(1/np.sqrt(np.sum(Phi**2, axis = 0))))

    I = np.zeros((patch_dim,batch_size))

    plt.ion()
    ahat_prev = None

    if not run_learning:

        # Tiling the image
        import pdb
        patch_per_dim = int(np.floor(imsize / sz))
        if not coeff_visualizer:
            batch_size = patch_per_dim**2

        I = np.zeros((patch_dim, batch_size))
        num_images = 200

        # Load dict from learning run
        Phi = scipy.io.loadmat('dict/Phi_%s_OC=%.1f_lambda=%.3f.mat' % (name, float(neurons)/patch_dim, lambdav))
        #print 'dict/Phi_%s_OC=%.1f_lambda=%.3f.mat' % (name, float(neurons)/patch_dim, lambdav)
        #Phi = scipy.io.loadmat('dict/Phi_IMAGES_DUCK_OC=2.0_lambda=0.01.mat')
        Phi = Phi['Phi']

        # **** Hack? Diff lambda for training vs. reconstructing ****
        lambdav = 0.02

        max_active = float(neurons * batch_size)

        #run_p = [(True, 120), (True, 90), (True, 60)] # Run params (use_prev?, iters)
        #run_p = [(True, 120), (True, 40), (True, 20), (False, 120), (False, 40), (False, 20)]
        #run_p = [(True, 120), (True, 40), (True, 20)]
        run_p = [(True, 10, 0.10)]
        #run_p = [
        #run_p = [(True, 60)]
        #run_p = [(True, 10), (True, 30), (False, 60)]
        #run_p = [(False, 80), (True, 5), (True, 3), (True, 1)]
        labels = []
        for (b,x,g) in run_p:
            if b == 1:
                labels += ["InitP (%d)" % x]
            else:
                labels += ["Init0 (%d)" % x]

        runs = len(labels)
        rcolor = [COLORS['red'], COLORS['green'], COLORS['blue'], COLORS['black'],
                  COLORS['yellow2'], COLORS['purple']]
        top_CC_AC=0

        for run in range(runs):
            # Record data
            MSE = [] # Mean Squared Error
            SNR = [] # Signal to Noise ratio
            AC  = np.zeros(num_images) # Active coefficients
            CC  = np.zeros(num_images) # Changing coefficients
            #DELTA = [0] * num_images
            ahat_prev = np.zeros((neurons, batch_size))
            ahat_prev_c = np.zeros((neurons, batch_size))
            lambdav = run_p[run][2]

            start = datetime.now()
            for t in range(num_images):
                i = 0
                for r in range(patch_per_dim):
                    for c in range(patch_per_dim):
                        r = c = 8
                        #if coeff_visualizer:
                            #r = np.floor(random.random() * patch_per_dim)
                            #c = np.floor(random.random() * patch_per_dim)
                        rr = r * sz
                        cc = c * sz
                        I[:,i] = np.reshape(IMAGES[rr:rr+sz, cc:cc+sz, t], patch_dim, 1)
                        if coeff_visualizer:
                            break
                        i = i + 1

                if run_p[run][0]:
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
                #DELTA[t] = delta
            elapsed = (datetime.now() - start).seconds

            plt.subplot(231)
            plt.xlabel('Time (steps)', fontdict={'fontsize':12})
            plt.ylabel('MSE', fontdict={'fontsize':12})
            plt.axis([0, num_images, 0.0, max(MSE) * 1.1])
            plt.plot(range(num_images), MSE, color=rcolor[run], label=labels[run])
            lg = plt.legend(bbox_to_anchor=(-0.6 , 0.40), loc=2, fontsize=10)
            lg.draw_frame(False)

            top_CC_AC = max(max(CC),max(AC),top_CC_AC) * 1.1
            plt.subplot(232)
            plt.xlabel('Time (steps)', fontdict={'fontsize':12})
            plt.ylabel('# Active Coeff', fontdict={'fontsize':12})
            plt.axis([0, num_images, 0, top_CC_AC])
            plt.plot(range(num_images), AC, color=rcolor[run])

            plt.subplot(233)
            plt.xlabel('Time (steps)', fontdict={'fontsize':12})
            plt.ylabel('# Changed Coeff', fontdict={'fontsize':12})
            plt.axis([0, num_images, 0, top_CC_AC])
            plt.plot(range(num_images), CC, color=rcolor[run])

            plt.subplot(234)
            plt.xlabel('Time (steps)', fontdict={'fontsize':12})
            plt.ylabel('SNR (dB)', fontdict={'fontsize':12})
            plt.axis([0, num_images, 0.0, 22])
            plt.plot(range(num_images), SNR, color=rcolor[run], label=labels[run])
            #lg = plt.legend(fontsize=10)
            lg = plt.legend(bbox_to_anchor=(-0.6 , 0.60), loc=2, fontsize=10)
            lg.draw_frame(False)

            #plt.subplot(234)
            #plt.xlabel('Time (steps)', fontdict={'fontsize':12})
            #plt.ylabel('# Delta in LCA', fontdict={'fontsize':12})
            #plt.axis([0, num_images, 0, top])
            #plt.plot(range(num_images), DELTA, color=rcolor[run])

            # % plots
            top_p = 100 * top_CC_AC / max_active
            plt.subplot(235)
            plt.xlabel('Time (steps)', fontdict={'fontsize':12})
            plt.ylabel('% Active Coeff', fontdict={'fontsize':12})
            plt.axis([0, num_images, 0, top_p])
            plt.plot(range(num_images), 100 * AC / max_active, color=rcolor[run])

            plt.subplot(236)
            plt.xlabel('Time (steps)', fontdict={'fontsize':12})
            plt.ylabel('% Changed Coeff', fontdict={'fontsize':12})
            plt.axis([0, num_images, 0, top_p])
            plt.plot(range(num_images), 100 * CC / max_active, color=rcolor[run])

        plt.suptitle("DATA=%s, LAMBDAV=%.2f, IMG=%dx%d, PAT=%dx%d, DICT=%d, PAT/IMG=%d ELAP=%d" %
                        (name, lambdav, imsize, imsize, sz, sz, neurons, patch_per_dim ** 2, elapsed), fontsize=18)
        plt.show(block=True)

        return False



    start = datetime.now()
    for t in range(num_trials):
        # Choose a random image
        imi = np.ceil(num_images * random.uniform(0, 1))

        for i in range(batch_size):
            r = border + np.ceil((imsize-sz-2*border) * random.uniform(0, 1))
            c = border + np.ceil((imsize-sz-2*border) * random.uniform(0, 1))

            I[:,i] = np.reshape(IMAGES[r:r+sz, c:c+sz, imi-1], patch_dim, 1)

        # Coefficient Inference
        ahat = sparsify(I, Phi, lambdav)

        # Calculate Residual
        R = I-np.dot(Phi, ahat)

        # Update Basis Functions
        dPhi = get_eta(t, batch_size) * (np.dot(R, ahat.T))
        Phi = Phi + dPhi
        Phi = np.dot(Phi, np.diag(1/np.sqrt(np.sum(Phi**2, axis = 0))))

        # Plot every 100 iterations
        if np.mod(t,100) == 0:
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

            #for t in range(num_images):
                #plt.imshow(IMAGES[:,:,t], cmap=cm.Greys_r, interpolation="nearest")
                #plt.draw()
                #time.sleep(4)
            plt.imshow(image, cmap=cm.Greys_r, interpolation="nearest")
            plt.draw()

        ahat_prev = ahat

    scipy.io.savemat('Phi_%s_OC=%.1f_lambda=%.2f' % (name, float(neurons)/patch_dim, lambdav),
                    {'Phi':Phi})
    plt.show()
    return Phi

def sparsify(I, Phi, lambdav, eta=0.05, ahat_prev=None, num_iterations=80):
    """
    LCA Inference.
    I: Image batch (dim x batch)
    Phi: Dictionary (dim x dictionary element)
    lambdav: Sparsity coefficient
    eta: Update rate
    """
    batch_size = np.shape(I)[1]

    (N, M) = np.shape(Phi)
    sz = np.sqrt(N)

    b = np.dot(Phi.T, I)
    G = np.dot(Phi.T, Phi) - np.eye(M)

    if run_learning:
        l = 0.5 * np.max(np.abs(b), axis = 0)
    else:
        #l = 0.1 * np.max(np.abs(b), axis = 0)
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
        plt.axis([0, M, 0, 1.05])

        plt.subplot(312)
        plt.imshow(np.reshape(I[:,0], (sz, sz)),cmap = cm.binary, interpolation='nearest')

        plt.subplot(311)
        recon = np.dot(Phi, a)
        plt.imshow(np.reshape(recon, (sz, sz)),cmap = cm.binary, interpolation='nearest')

        plt.draw()
        plt.show()
        time.sleep(1)

    for t in range(num_iterations):
        u = eta * (b - np.dot(G, a)) + (1 - eta) * u
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

    if coeff_visualizer:
        plt.close()

    return a

def g(u,theta,thresh_type='hard'):
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
