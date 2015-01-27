import numpy as np
import h5py
import scipy.io

t = 150 # frames
#t = 5
x = range(0,t)

#a,n = 0.8, 5.0   # Chosen for e-2 at 30Hz
#a,n = 0.9, 10.0  # Sharper...
a,n = 0.7, 3.333 # Sharper...
filt = (1/n) * np.power(a, x)

name = 'IMAGES_DUCK_LONG'

if name == 'IMAGES_DUCK_LONG':
    # Load data, have to use h5py because had to use v7.3 because .mat is so big.
    f = h5py.File('mat/%s.mat' % name, 'r',)
    IMAGES = np.array(f.get(name))
    IMAGES = np.swapaxes(IMAGES, 0, 2) # v7.3 reorders for some reason, or h5?
else:
    IMAGES = scipy.io.loadmat('mat/%s.mat' % name)[name]

NIMAGES = np.empty((IMAGES.shape[0], IMAGES.shape[1], IMAGES.shape[2] - t))
print 'Beginning Convolution', IMAGES.shape
for i in range(IMAGES.shape[0]):
    print 'i=%d' % i
    for j in range(IMAGES.shape[1]):
        l = IMAGES.shape[2]
        NIMAGES[i][j] = np.convolve(IMAGES[i][j], filt)[t:l] # Cut out first and last *t* frames

nname = '%s_SMOOTH_%.1f' % (name, a)
scipy.io.savemat('mat/%s' % nname, {nname:NIMAGES})

