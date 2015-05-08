import pdb
import scipy.io
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from showbfs import showbfs

#init_Phi = 'Phi_193_37.0.mat'
init_Phi = 'Phi_472/Phi_472_0.0.mat'
init_Phi = 'Phi_473/Phi_473_0.1.mat'
init_Phi = 'Phi_475/Phi_475_0.0.mat'
Phi = scipy.io.loadmat('dict/%s' % init_Phi)['Phi']

#idx =
#arr = [np.array([Phi[:,j]]) for j in [20, 451, 582, 651, 710, 786, 908]]
arr = [np.array([Phi[:,j]]) for j in [327, 494]]
arr = [np.array([Phi[:,j]]) for j in [462, 255]]
arr = [np.array([Phi[:,j]]) for j in [365, 152]]

group = np.concatenate(arr).T
showbfs(group)
#plt.title('i=%d, j=%d Dot_Prod=%.3f' % (327, 494, np.dot(Phi[:,0].T, Phi[:, 1])))
plt.title('i=%d, j=%d Dot_Prod=%.3f' % (327, 494, np.dot(Phi[:,1].T, Phi[:, 0])))
plt.show()
