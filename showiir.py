import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from showbfs import showbfs

idx = 14
log = 'iir_LOG_%d.npz' % idx
d = np.load('iir_dict/%s' % log)

Phi = d['Phi']
B = d['B']
M = d['M']

showbfs(Phi)
plt.title('%s - Phi' % log)
plt.show(block=True)

plt.imshow(B, interpolation='nearest', norm=matplotlib.colors.Normalize(-0.1,0.1,True))
plt.colorbar()
plt.title('%s - B' % log)
plt.show(block=True)

plt.imshow(M, interpolation='nearest', norm=matplotlib.colors.Normalize(-0.01,0.01,True))
plt.colorbar()
print 'max m', np.max(M)
plt.title('%s - M' % log)
plt.show(block=True)
