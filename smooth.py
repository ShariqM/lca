import numpy as np
import scipy.io
#t = 150
t = 5
x = range(0,t)

a = 0.8 # Chosen for e-2 at 30Hz
filt = np.power(a, x)

name = 'IMAGES_DUCK_SHORT'
IMAGES = scipy.io.loadmat('mat/%s.mat' % name)['IMAGES_DUCK']
print 'Beginning Convolution', IMAGES.shape
for i in range(IMAGES.shape[0]):
    for j in range(IMAGES.shape[1]):
        l = IMAGES.shape[2]
        IMAGES[i][j] = np.convolve(IMAGES[i][j], filt)[:l]

scipy.io.savemat('mat/IMAGES_DUCK_SHORT_SMOOTH', {'IMAGES':IMAGES})

