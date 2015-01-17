import numpy as np
import scipy.io
#t = 150
t = 5
x = range(0,t)

a = 0.8 # Chosen for e-2 at 30Hz
filt = np.power(a, x)

name = 'IMAGES_DUCK'
IMAGES = scipy.io.loadmat('mat/%s.mat' % name)[name]
print 'Beginning Convolution'
for i in range(IMAGES.shape[0]):
    for j in range(IMAGES.shape[1]):
        l = IMAGES.shape[2]
        IMAGES[i][j] = np.convolve(IMAGES[i][j], filt)[:l]

scipy.io.savemat('IMAGES_DUCK_SMOOTH', {'IMAGES':IMAGES})

