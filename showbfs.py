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

def showbfs(Phi, idx=-1):
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

    #plt.imshow(arr, cmap = cm.binary, interpolation='nearest')
    plt.imshow(arr, cmap = cm.binary, interpolation='none')
    plt.title('Phi_%d' % idx)
    plt.draw()

if __name__ == "__main__":
    i = args.dict_idx
    name = 'Phi_197_1.9.mat'
    name = 'Phi_399_0.4.mat'
    name = 'Phi_463/Phi_463_0.3.mat'
    #name = 'Phi_600/Phi_600_1.3.mat'
    #name = 'Phi_601/Phi_601_0.3.mat'
    #name = 'Phi_602/Phi_602_0.2.mat'
    #name = 'Phi_207_0.8.mat'
    #name = 'Phi_208_0.7.mat'
    #name = 'Phi_209_0.1.mat'
    #name = 'Phi_198_0.6.mat'

    #idx = int(name.split('_')[1])
    idx = -1

    Phi = scipy.io.loadmat('dict/%s' % name)['Phi']
    showbfs(Phi, idx)
    plt.show()
