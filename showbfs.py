import scipy.io
import time
import numpy
from numpy import reshape, zeros, ones
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from math import floor, ceil, sqrt
import pprint

def showbfs(Phi):
    (L, M) = Phi.shape # L = pixels of image, M = num images

    sz = sqrt(L) # sz of one side of image
    print sz
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
    plt.draw()

#Phi = scipy.io.loadmat('dict/Phi_IMAGES_DUCK_OC=4.0_lambda=0.01.mat')['Phi']
#Phi = scipy.io.loadmat('dict/Phi_IMAGES_DUCK_OC=2.0_lambda=0.01.mat')['Phi']

#Phi = scipy.io.loadmat('dict/Phi_IMAGES_DUCK_OC=4.0_lambda=0.007.mat')['Phi']
Phi = scipy.io.loadmat('dict/Phi_3.mat')['Phi']
showbfs(Phi)
plt.show()

# Show image
#Phi = scipy.io.loadmat('mat/IMAGES.mat')['IMAGES']
#Phi = scipy.io.loadmat('IMAGES_GARB_SMOOTH.mat')['IMAGES']

#Phi = scipy.io.loadmat('IMAGES_DUCK_SMOOTH.mat')['IMAGES']
#Phi = scipy.io.loadmat('mat/IMAGES_DUCK.mat')['IMAGES_DUCK']
#Phi = scipy.io.loadmat('mat/IMAGES_DUCK_SHORT_NO_VAR.mat')['IMAGES_DUCK']








Phi = scipy.io.loadmat('mat/IMAGES_DUCK_SHORT.mat')['IMAGES_DUCK']
import matplotlib

print Phi.shape
plt.ion()
for i in range(Phi.shape[2]):
    #plt.imshow(Phi[:,:,i*10], cmap = cm.binary, interpolation='nearest')
    plt.imshow(Phi[:,:,i], norm=matplotlib.colors.Normalize(-1,1,True), cmap = cm.binary)
    plt.title('%d Var=%.4f' % (i, Phi[:,:,i].var().mean()))
    plt.draw()
    plt.show()
    time.sleep(0.01)
    plt.clf()
