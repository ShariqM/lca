import pdb
import scipy.io
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from showbfs import showbfs

init_Phi = 'Phi_193_37.0.mat'
Phi = scipy.io.loadmat('dict/%s' % init_Phi)['Phi']

idx =
arr = [np.array([Phi[:,j]]) for j in [20, 451, 582, 651, 710, 786, 908]]

group = np.concatenate(arr).T
showbfs(group)
plt.title('i=%d' % 20)
plt.show()
