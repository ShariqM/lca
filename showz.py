import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.io
phi_name = 'Phi_601/Phi_601_0.3'
Z = scipy.io.loadmat('dict/%s' % phi_name)['Z']
plt.imshow(Z, interpolation='nearest', norm=matplotlib.colors.Normalize(-1,1,True))
plt.colorbar()
plt.show(block=True)

