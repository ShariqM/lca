import matplotlib.pyplot as plt
import numpy as np
from numpy.random import rand
import pdb

activity_log = np.load('coeff_S193__initP20_100t.npy')
print 'max activity log', np.max(activity_log[:,0,:])
print 'mean activity log', np.mean(activity_log[:,0,:])

cov = np.cov(np.abs(activity_log[:,0,:]))
neurons = cov.shape[0]

#threshold = np.mean(cov) + 5 * np.var(cov)
threshold = np.max(cov) * 0.1
arrs = []

for i in range(neurons):
    l = []
    for j in range(neurons):
        if j < i:
            continue
        if cov[i,j] > threshold:
            l.append(j)
    if len(l) > 0:
        arrs.append(l)
        print 'Neuron %d is friends with ' % i, l


import scipy.io
import matplotlib.pyplot as plt
from showbfs import showbfs

init_Phi = 'Phi_193_37.0.mat'
Phi = scipy.io.loadmat('dict/%s' % init_Phi)['Phi']

for l in arrs:
    arr = [np.array([Phi[:,j]]) for j in l]
    group = np.concatenate(arr).T
    showbfs(group)
    plt.title('i=%d' % 20)
    plt.show()
