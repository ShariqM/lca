import numpy as np
import scipy.io
import pdb

data_name = 'IMAGES_DUCK_SHORT'
images = scipy.io.loadmat('mat/%s.mat' % data_name)
images = images[data_name]

(imsz, imsz, timepoints) = images.shape
nimages = np.zeros((imsz, imsz, timepoints * 10))

for i in range(timepoints - 1):
    for j in range(10):
        nimages[:,:, i*10 + j] = (1 - (j/10.0)) * images[:,:,i] + (j/10.0) * images[:,:,i+1]
        pdb.set_trace()

ndata_name = data_name + '_INTERP'
scipy.io.savemat('mat/%s' % ndata_name, {ndata_name:nimages})
