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
import pdb

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("-d", "--idx", dest="dict_idx", default=1,
                    type=int, help="Dictionary index to show")
args = parser.parse_args()


def showbfs(Phi, eta=-1.0):
    L,M = Phi.shape

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
    #Phi = scipy.io.loadmat('dict/Phi_.mat/Phi_6/Phi_67/Phi_67_1.2.mat')['Phi']
    #Phi = scipy.io.loadmat('dict/Phi_73_red.mat')['Phi']
    #Phi = scipy.io.loadmat('dict/Phi_75_100.0.mat')['Phi']
    #Phi = scipy.io.loadmat('dict/Phi_90/Phi_90_3.1')['Phi']
    #Phi = scipy.io.loadmat('dict/Phi_101/Phi_101_20.0')['Phi']
    #Phi = scipy.io.loadmat('dict/Phi_110/Phi_110_100.0')['Phi']
    #Phi = scipy.io.loadmat('dict/Phi_117/Phi_117_1.6.mat')['Phi']
    Phi = scipy.io.loadmat('dict/Phi_193_37.0.mat')['Phi']
    pdb.set_trace()
    #Phi = scipy.io.loadmat('dict/Phi_IMAGES_DUCK_OC=4.0_lambda=0.007.mat')['Phi']
    showbfs(Phi)
    plt.show()
