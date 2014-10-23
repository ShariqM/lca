import scipy.io
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import cm
import time
from datetime import datetime
import fista

def get_eta(t,batch_size):
    start = 2000
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

def sparsenet(patch_dim=64, neurons=128, lambdav=0.05, eta=6.0, num_trials=3000, batch_size=100, border=4, inference='lca', run_learning=True):
    """
    N: # Inputs
    M: # Outputs
    lambdav: Sparsity Constraint
    eta: Learning Rate
    num_trials: Learning Iterations
    batch_size: Batch size per iteration
    border: Border when extracting image patches
    Inference: 'lca' or 'fista'
    """
    name='IMAGES_FOREMAN'
    IMAGES = scipy.io.loadmat('./%s.mat' % name)[name]
    (imsize, imsize, num_images) = np.shape(IMAGES)

    sz = np.sqrt(patch_dim)
    eta = eta / batch_size

    # Initialize basis functions
    Phi = np.random.randn(patch_dim, neurons)
    Phi = np.dot(Phi, np.diag(1/np.sqrt(np.sum(Phi**2, axis = 0))))

    plt.ion()

    if not run_learning:
        # Tiling the image
        patch_per_dim = int(np.floor(imsize / sz))
        batch_size = patch_per_dim**2

        I = np.zeros((patch_dim, batch_size))
        num_images = 200

        # Load dict from learning run
        Phi = scipy.io.loadmat('Phi_IMAGES_FOREMAN_lambda=0.05.mat')['Phi']

        # Record data
        MSE = [] # Mean Squared Error
        CA  = np.zeros(num_images) # Active coefficients
        CF  = np.zeros(num_images) # Changing coefficients
        DELTA = [0] * num_images
        ahat_prev = np.zeros((neurons, batch_size))

        max_active = float(neurons * batch_size)

        start = datetime.now()
        for t in range(num_images):
            i = 0
            for r in range(patch_per_dim):
                for c in range(patch_per_dim):
                    rr = r * sz
                    cc = c * sz
                    I[:,i] = np.reshape(IMAGES[rr:rr+sz, cc:cc+sz, t], patch_dim, 1)
                    i = i + 1

            #ahat, delta = sparsify(I, Phi, lambdav)
            ahat, delta = sparsify(I, Phi, lambdav, ahat_prev)

            # Calculate Residual Error
            R = I - np.dot(Phi, ahat)
            MSE.append((R ** 2).mean())

            ahat[np.abs(ahat) > lambdav/1000.0] = 1
            CA[t] = np.sum(ahat)
            CF[t] = np.sum(ahat!=ahat_prev)
            DELTA[t] = delta
            ahat_prev = ahat
        elapsed = (datetime.now() - start).seconds


        plt.subplot(231)
        plt.xlabel('Time (steps)', fontdict={'fontsize':12})
        plt.ylabel('MSE', fontdict={'fontsize':12})
        plt.axis([0, num_images, 0.0, max(MSE) * 1.1])
        plt.plot(range(num_images), MSE)


        top = max(max(CF),max(CA)) * 1.1
        plt.subplot(232)
        plt.xlabel('Time (steps)', fontdict={'fontsize':12})
        plt.ylabel('# Active Coeff', fontdict={'fontsize':12})
        plt.axis([0, num_images, 0, top])
        plt.plot(range(num_images), CA)

        plt.subplot(233)
        plt.xlabel('Time (steps)', fontdict={'fontsize':12})
        plt.ylabel('# Changed Coeff', fontdict={'fontsize':12})
        plt.axis([0, num_images, 0, top])
        plt.plot(range(num_images), CF)

        plt.subplot(234)
        plt.xlabel('Time (steps)', fontdict={'fontsize':12})
        plt.ylabel('# Delta in LCA', fontdict={'fontsize':12})
        #plt.axis([0, num_images, 0, top])
        plt.plot(range(num_images), DELTA)

        # % plots
        top_p = 100 * top / max_active
        plt.subplot(235)
        plt.xlabel('Time (steps)', fontdict={'fontsize':12})
        plt.ylabel('% Active Coeff', fontdict={'fontsize':12})
        plt.axis([0, num_images, 0, top_p])
        plt.plot(range(num_images), 100 * CA / max_active)

        plt.subplot(236)
        plt.xlabel('Time (steps)', fontdict={'fontsize':12})
        plt.ylabel('% Changed Coeff', fontdict={'fontsize':12})
        plt.axis([0, num_images, 0, top_p])
        plt.plot(range(num_images), 100 * CF / max_active)

        plt.suptitle("LCA Analysis, IMG=%dx%d, PAT=%dx%d, DICT=%d, PAT/IMG=%d ELAP=%d" %
                        (imsize, imsize, sz, sz, neurons, patch_per_dim ** 2, elapsed), fontsize=18)
        plt.show(block=True)

        return False

    I = np.zeros((patch_dim,batch_size))

    for t in range(num_trials):

        # Choose a random image
        imi = np.ceil(num_images * random.uniform(0, 1))

        for i in range(batch_size):
            r = border + np.ceil((imsize-sz-2*border) * random.uniform(0, 1))
            c = border + np.ceil((imsize-sz-2*border) * random.uniform(0, 1))

            I[:,i] = np.reshape(IMAGES[r:r+sz, c:c+sz, imi-1], patch_dim, 1)

            # Coefficient Inference
            if inference == 'lca':
                ahat = sparsify(I, Phi, lambdav)
            elif inference == 'fista':
                ahat = fista.fista(I, Phi, lambdav, max_iterations=50)
            else:
                print "Invalid inference option"
            return

            # Calculate Residual Error
            R = I-np.dot(Phi, ahat)

            # Update Basis Functions
            dPhi = get_eta(t, batch_size) * (np.dot(R, ahat.T))
            Phi = Phi + dPhi
            Phi = np.dot(Phi, np.diag(1/np.sqrt(np.sum(Phi**2, axis = 0))))

            # Plot every 100 iterations
            if np.mod(t, 20) == 0:
                print "Iteration %d, Error = %f " % (t, np.sum(R))
                side = np.sqrt(neurons)
                image = np.zeros((sz*side+side,sz*side+side))
                for i in range(side.astype(int)):
                    for j in range(side.astype(int)):
                        patch = np.reshape(Phi[:,i*side+j],(sz,sz))
                        patch = patch/np.max(np.abs(patch))
                        image[i*sz+i:i*sz+sz+i,j*sz+j:j*sz+sz+j] = patch

            plt.imshow(image, cmap=cm.Greys_r, interpolation="nearest")
            plt.draw()

    #scipy.io.savemat('Phi_%s_lambda=%.2f' % (name,lambdav), {'Phi':Phi})
    plt.show(block=True) # Display
    return Phi, ahat

def sparsify(I, Phi, lambdav, ahat_prev=None, eta=0.1, num_iterations=125):
    """
    LCA Inference.
    I: Image batch (patch x batch)
    Phi: Dictionary (patch x neurons)
    lambdav: Sparsity coefficient
    eta: Update rate
    """
    batch_size = np.shape(I)[1]

    (N, M) = np.shape(Phi)
    sz = np.sqrt(N)

    b = np.dot(Phi.T, I)
    G = np.dot(Phi.T, Phi) - np.eye(M)

    l = 0.1 * np.max(np.abs(b), axis=0) # Threshold per patch
    if ahat_prev is not None:
        #  u - neuronsx384, l - 384x1
        u = np.dot(ahat_prev, np.diag(l))
    else:
        u = np.zeros((M,batch_size))

    a = g(u, l)

    init_sum = (b - np.dot(G,a)).sum()
    for t in range(num_iterations):
        #tmp = b - np.dot(G,a)
        #print '\tTMP MEAN:%f SUM:%f ' % (tmp.mean(), tmp.sum())
        #u = eta * tmp + (1 - eta) * u
        u = eta * (b - np.dot(G, a)) + (1 - eta) * u
        a = g(u, l)

        #l = 0.95 * l
        #l[l < lambdav] = lambdav

    print 'init_sum: %f vs. %f :final_sum' % (init_sum, (b - np.dot(G,a)).sum())
    return a, init_sum - (b - np.dot(G,a)).sum()

# g:  Hard threshold. L0 approximation
def g(u,theta):
    """
    LCA threshold function
    u: coefficients
    theta: threshold value
    """
    a = u;
    a[np.abs(a) < theta] = 0
    return a

sparsenet(run_learning=False)
