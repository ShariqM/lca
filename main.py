import numpy as np
import scipy.io as io
from math import floor
from random import random

img_file = 'IMAGES.mat'
IMAGES = io.loadmat(img_file)['IMAGES']
(imsize, imsize, num_images) = IMAGES.shape

sz = 16
N = sz * sz # Pixels
M = N       # Dictionary elements

phi = np.array((M,N))
a = np.zeros(M)

def reconstruct(phi, a):
    return np.dot(a.T, phi)

def get_subimages(batch_size):
    BUFF = 4
    imi = floor(num_images*random())
    I = numpy.array(N, batch_size)
    for i in range(batch_size)
        r = BUFF + floor((imgsize-sz-2*BUFF) * random())
        c = BUFF + floor((imgsize-sz-2*BUFF) * random())
        I[:,i] = np.reshape(IMAGES[r:(r+sz-1), c:(c+sz-1), imi], N, 1)
    return I

batch_size = 100
I = get_subimages(batch_size)

    # MxN x NxBS (batch_size)
b = np.dot(phi, I)
G = np.dot(phi, phi.T) - np.identity(M)

u = np.zeros(M)
l = 0.5 * np.amax(abs(b))
def thres(u_m):
    if u_m >= l:
        return u_m
    return 0
vecthres = np.vectorize(thres)
a = vecthres(u)

tau = 10 # ms

num_iterations = 200;
eta = 0.1;

for i in range(num_iterations):
    u = eta * (b - np.dot(G, a)) + (1-eta) * u
    a = vecthres(u)

    # l = 0.98*l  # Hm...

#while True:
    # Mx1 =    MxN     Nx1
    #b = np.dot(phi[0], s)

    # 1x1 =             1x1 - 1x1 - (       NxM x MxN
    #du_m = (1.0/tau) * (b_m - u_m - (np.dot(phi.T, phi) - np.identity) # OLD

    # Mx1 =           Mx1 - Mx1 -       (       (MxN  x NxM      - MxM)         x   Mx1        )
    #du = (1.0/tau) * ( b  -  u  - np.dot((np.dot(phi, phi.T) - np.identity(M)), vecthres(u)))

    #u += du
