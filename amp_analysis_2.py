import pdb
import scipy.io
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from showbfs import showbfs

# LCA
#Phi, gs = scipy.io.loadmat('dict/Phi_204_15.4.mat')['Phi'],  4
#Phi, gs = scipy.io.loadmat('dict/Phi_199_100.0.mat')['Phi'], 2
#Phi, gs = scipy.io.loadmat('dict/Phi_201_17.0.mat')['Phi'], 2

# vLCA
#Phi, gs = scipy.io.loadmat('dict/Phi_205_36.6.mat')['Phi'],  16
#Phi, gs = scipy.io.loadmat('dict/Phi_198_0.6.mat')['Phi'], 8
Phi, gs = scipy.io.loadmat('dict/Phi_197_0.8.mat')['Phi'], 4
#Phi, gs = scipy.io.loadmat('dict/Phi_194_1.2.mat')['Phi'], 2

group_sparse = gs

for i in range(Phi.shape[1]):
    size = Phi.shape[1]/group_sparse

    arr = [np.array([Phi[:,j]]) for j in range(i, Phi.shape[1], size)]
    group = np.concatenate(arr).T
    showbfs(group)
    plt.title('i=%d' % i)
    plt.show()
