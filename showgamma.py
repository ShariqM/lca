import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.io
import numpy as np


G_init = 'AA_LOG_83'
Gam = np.copy(scipy.io.loadmat('activity/%s' % G_init)['Gam'])

for i in range(Gam.shape[2]):
    plt.imshow(Gam[:,:,i], interpolation='nearest', norm=matplotlib.colors.Normalize(-1,1,True))
    plt.title("%d/%d" % (i, Gam.shape[2]))
    plt.colorbar()
    plt.show(block=True)

