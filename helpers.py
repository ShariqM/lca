import socket
import numpy as np
from theano import *
import theano.tensor as T

# Theano Matrix Multiplication Optimization
if socket.gethostname() == 'redwood2':
    Gv = T.fmatrix('G')
    av = T.fmatrix('a')
    o  = T.dot(Gv, av)
    tdot = theano.function([Gv, av], o, allow_input_downcast=True)
else:
    tdot = np.dot

def get_images(image_data_name):
    if 'LONG' or '120' in image_data_name:
        f = h5py.File('mat/%s.mat' % image_data_name, 'r',) # Need h5py for big file
        IMAGES = np.array(f.get(image_data_name))
        IMAGES = np.swapaxes(IMAGES, 0, 2) # v7.3 reorders for some reason, or h5?
    else:
        IMAGES = scipy.io.loadmat('mat/%s.mat' % image_data_name)[image_data_name]
    return IMAGES

def check_activity_old(b, G, u, a):
    if np.sum(np.abs(u)) > 0000: # Coeff explosion check
        print 'Activity Explosion!!!'
        print 'Data:'
        x = np.abs(tdot(G,a))
        print 'b (sum, min, max)', np.sum(b), np.min(b), np.max(b)
        print 'f(G,a) (sum, min, max)', np.sum(x), np.min(x), np.max(x)
        print 'u (sum, min, max)', np.sum(u), np.min(u), np.max(u)

def check_activity(b, G, u, a):
    if np.sum(np.abs(u)) > 20000: # Coeff explosion check
        print 'Activity Explosion!!!'
        print 'Data:'
        Ga = np.abs(tdot(G,a))

        bc = np.abs(np.copy(b))
        Gac = np.abs(np.copy(Ga))
        uc = np.abs(np.copy(u))
        print 'bc (sum, min, max)',np.sum(bc), np.min(bc), np.max(bc)
        print 'Gac (sum, min, max)',np.sum(Gac), np.min(Gac), np.max(Gac)
        print 'uc (sum, min, max)',np.sum(uc), np.min(uc), np.max(uc)

def check_activity_m(b, Zb, Ga, ZGa, u):
    if np.sum(np.abs(u)) > 20000: # Coeff explosion check
        print 'Activity Explosion!!!'
        print 'Data:'
        bc = np.abs(np.copy(b))
        Zbc = np.abs(np.copy(Zb))
        Gac = np.abs(np.copy(Ga))
        ZGac = np.abs(np.copy(ZGa))
        uc = np.abs(np.copy(u))
        print 'bc (sum, min, max)',np.sum(bc), np.min(bc), np.max(bc)
        print 'Zbc (sum, min, max)',np.sum(Zbc), np.min(Zbc), np.max(Zbc)
        print 'Gac (sum, min, max)',np.sum(Gac), np.min(Gac), np.max(Gac)
        print 'ZGac (sum, min, max)',np.sum(ZGac), np.min(ZGac), np.max(ZGac)
        print 'uc (sum, min, max)',np.sum(uc), np.min(uc), np.max(uc)



# Simulated Annealing functions
def get_eta(t, neurons, runtype, batch_size):
    if neurons < 300:
        start = 500
        inc = 500
    else:
        start = 1000
        inc = 1000
    if t < start:
        return 6.0/batch_size
    if t < start + 1*inc:
        return 3.0/batch_size
    if t < start + 2*inc:
        return 1.0/batch_size
    if t < start + 3*inc:
        return 0.5/batch_size
    if t < start + 4*inc:
        return 0.25/batch_size
    if t < start + 5*inc:
        return 0.125/batch_size
    if t < start + 6*inc:
        return 0.06/batch_size
    if t < start + 7*inc:
        return 0.03/batch_size
    if t < start + 8*inc:
        return 0.015/batch_size
    return 0.01/batch_size

def get_veta(t, neurons, runtype, batch_size):
    if neurons < 300:
        start = 500
        inc = 500
    else:
        start = 2000
        inc = 1000

    if t < start:
        return 6.0/batch_size
    if t < start + 1*inc:
        return 3.0/batch_size
    if t < start + 2*inc:
        return 1.5/batch_size
    if t < start + 3*inc:
        return 0.75/batch_size
    if t < start + 4*inc:
        return 0.375/batch_size
    if t < start + 5*inc:
        return 0.18/batch_size
    return 0.10/batch_size
