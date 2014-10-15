import numpy as np
import scipy.io as io
import time
import matplotlib.pyplot as plt
from math import floor,sqrt
from random import random
from showbfs import showbfs

def reconstruct(phi, a):
    return np.dot(a.T, phi)

def get_subimages(batch_size):
    BUFF = 4
    imi = floor(num_images*random())
    I = np.zeros((N, batch_size))
    for i in range(batch_size):
        r = BUFF + floor((imsize-sz-2*BUFF) * random())
        c = BUFF + floor((imsize-sz-2*BUFF) * random())
        I[:,i] = np.reshape(IMAGES[r:(r+sz), c:(c+sz), imi], N, 1)
    return I

def sparsify(I, phi, lambdah):
    # MxBS = MxN x NxBS (batch_size)
    b = np.dot(phi.T, I) # sometimes get nan...
    G = np.dot(phi.T, phi) - np.eye(M)

    u = np.zeros((M,batch_size))
    l = 0.5 * np.max(np.abs(b), axis=0)
    a = g(u,l)

    num_iterations = 125;
    eta = 0.1;

    for i in range(num_iterations):
        u = eta * (b - np.dot(G, a)) + (1 - eta) * u
        a = g(u, l)

        l = 0.95 * l
        l[l < lambdah] = lambdah

    return a

def g(u,l):
  a = u;
  a[np.abs(a) < l] = 0
  return a

img_file = 'IMAGES_Z.mat'
IMAGES = io.loadmat(img_file)['IMAGES']
(imsize, imsize, num_images) = IMAGES.shape

sz = 8
N = sz * sz # Pixels
M = 2 * N       # Dictionary elements

phi = np.random.randn(N,M)
phi = np.dot(phi, np.diag(1/np.sqrt(np.sum(phi**2, axis = 0))))
a = np.zeros(M)
plt.ion()
showbfs(phi)

batch_size = 100
eta = 6.0/batch_size
lambdah = 0.1;
showbfs(phi)

def get_eta(t):
    if t < 1000:
        return 6.0/batch_size
    if t < 1500:
        return 3.0/batch_size
    if t < 2000:
        return 1.0/batch_size
    if t < 2500:
        return 0.5/batch_size
    if t < 3000:
        return 0.25/batch_size
    if t < 3500:
        return 0.10/batch_size
    return 0.10/batch_size

for t in range(10000):
    eta = get_eta(t)
    # Random images
    I = get_subimages(batch_size)

    # Coefficient Inference
    ahat = sparsify(I, phi, lambdah)

    # Error = Nxbatch_size - NxM * Mxbatch_size
    R = I - np.dot(phi, ahat)

    # Update phi
    dphi = eta * np.dot(R, ahat.T)
    phi = phi + dphi

    # Normalize
    phi = np.dot(phi, np.diag(1/np.sqrt(np.sum(phi**2, axis = 0))))
    if t % 20 == 0:
        print 'Trial %d | Err=%f' % (t, np.sum(R))
        showbfs(phi)

print "Coeff", a
plt.show()
time.sleep(10000)

'''
s = np.sum(abs(np.dot(phi.T,phi)), 0) # Added abs
phi = np.dot(phi, np.diag(1.0 / np.sqrt(s)))
'''
