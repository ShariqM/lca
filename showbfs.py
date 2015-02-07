import scipy.io
import numpy as np
import time
import numpy
from numpy import reshape, zeros, ones
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from math import floor, ceil, sqrt
import pprint
import h5py
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("-d", "--idx", dest="dict_idx", default=1,
                    type=int, help="Dictionary index to show")
args = parser.parse_args()


def showbfs(Phi):
    (patch_dim, neurons) = Phi.shape

    sz = np.sqrt(patch_dim)

    side = np.sqrt(neurons)
    image = np.zeros((sz*side+side,sz*side+side))
    for i in range(int(side)):
        for j in range(int(side)):
            patch = np.reshape(Phi[:,i*side+j],(sz,sz))
            patch = patch/np.max(np.abs(patch))
            image[i*sz+i:i*sz+sz+i,j*sz+j:j*sz+sz+j] = patch

    plt.imshow(image, cmap=cm.Greys_r, interpolation="nearest")
    plt.draw()
    plt.show()

def gg():

    sz = sqrt(L) # sz of one side of basis
    n = floor(sqrt(M)) # sz of one side of the grid of images
    m = ceil(M/n) # ceil for 1 extra
    buf = 1

    arr = ones(shape=(buf + n * (sz + buf), buf + m * (sz + buf)))

    for k in range(M):
        i = (k % n)
        j = floor(k/n)

        def index(x):
            return buf + x * (sz + buf)

        maxA=max(abs(Phi[:,k])) # RESCALE
        img = reshape(Phi[:,k], (sz, sz), order='C').transpose()/maxA
        arr[index(i):index(i)+sz, index(j):index(j)+sz] = img

    plt.imshow(arr, cmap = cm.binary, interpolation='nearest')
    plt.title("Phi_%d" % args.dict_idx)
    plt.draw()

if __name__ == "__main__":
    i = args.dict_idx
    #Phi = scipy.io.loadmat('dict/Phi_%d/Phi_%d_0.6.mat' % (i, i))['Phi']
    #Phi = scipy.io.loadmat('dict/Phi_%d/Phi_%d.mat' % (i, i))['Phi']
    #Phi = scipy.io.loadmat('dict/Phi_.mat/Phi_11.mat')['Phi']
    Phi = scipy.io.loadmat('dict/Phi_.mat/Phi_6/Phi_67/Phi_67_1.2.mat')['Phi']
    #Phi = scipy.io.loadmat('dict/Phi_IMAGES_DUCK_OC=4.0_lambda=0.007.mat')['Phi']
    showbfs(Phi)
    plt.show()
