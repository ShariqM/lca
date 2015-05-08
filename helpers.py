import socket
import pdb
import numpy as np
from theano import *
import theano.tensor as T

# Theano Matrix Multiplication Optimization
if socket.gethostname() == 'redwood2':
    Av = T.fmatrix('A')
    Bv = T.fmatrix('B')
    o2 = T.dot(Av, Bv)
    t2dot = theano.function([Av, Bv], o2, allow_input_downcast=True)
    tdot = t2dot

    Cv = T.fmatrix('C')
    Dv = T.fmatrix('D')
    Ev = T.fmatrix('E')
    o3 = T.dot(T.dot(Cv, Dv), Ev)
    t3dot = theano.function([Cv, Dv, Ev], o3, allow_input_downcast=True)

    Fv = T.fmatrix('F')
    Gv = T.fmatrix('G')
    Hv = T.fmatrix('H')
    Iv = T.fmatrix('I')
    o4 = T.dot(T.dot(Fv, Gv), T.dot(Hv, Iv))
    t4dot = theano.function([Fv, Gv, Hv, Iv], o4, allow_input_downcast=True)

    Jv = T.fmatrix('J')
    Kv = T.fmatrix('K')
    Lv = T.fmatrix('L')
    Mv = T.fmatrix('M')
    Nv = T.fmatrix('N')
    o5 = T.dot(T.dot(Jv, T.dot(Kv, Lv)), T.dot(Mv, Nv))
    t5dot = theano.function([Jv, Kv, Lv, Mv, Nv], o5, allow_input_downcast=True)

else:
    def dot_many(*args):
        A = []
        for a in args:
            A.append(a)
        return reduce(nump.dot, A)
    tdot = t2dot = t3dot = t4dot = t5dot = np.dot

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
        x = np.abs(t2dot(G,a))
        print 'b (sum, min, max)', np.sum(b), np.min(b), np.max(b)
        print 'f(G,a) (sum, min, max)', np.sum(x), np.min(x), np.max(x)
        print 'u (sum, min, max)', np.sum(u), np.min(u), np.max(u)

def check_activity(b, G, u, a):
    if np.sum(np.abs(u)) > 5000: # Coeff explosion check
        print 'Activity Explosion!!!'
        print 'Data:'
        Ga = np.abs(t2dot(G,a))

        #bc = np.abs(np.copy(b))
        #Gac = np.abs(np.copy(Ga))
        #uc = np.abs(np.copy(u))
        bc = b
        Gac = Ga
        uc = u
        print 'bc (sum, min, max)',np.sum(bc), np.min(bc), np.max(bc)
        print 'Gac (sum, min, max)',np.sum(Gac), np.min(Gac), np.max(Gac)
        print 'uc (sum, min, max)',np.sum(uc), np.min(uc), np.max(uc)
        print 'uc index', np.where(uc==np.min(uc)), np.where(uc==np.max(uc))
        return True

def check_activity_m(b, Zb, Ga, ZGa, u):
    if np.sum(np.abs(np.max(u))) > 20: # Coeff explosion check
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
        return True
    return False



# Simulated Annealing functions
def get_eta(t, neurons, runtype, batch_size):
    if neurons < 300:
        start = 300
        inc = 300
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
    if neurons < 200:
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

def get_zeta_2(t, neurons, runtype, batch_size):
    start = 2000
    inc = 1000

    eta = eta_start = 0.20

    for i in range(6):
        if t < start + i*inc:
            return eta/batch_size
        eta /= 2.0 # cut in half
    return eta/batch_size

# Move Right LCA
def get_zeta(t, neurons, runtype, batch_size):
    start = 2000
    inc = 1000

    eta = eta_start = 0.60

    for i in range(6):
        if t < start + i*inc:
            return eta/batch_size
        eta /= 2.0 # cut in half
    return eta/batch_size

# Move Right on TSC
def get_zeta_g(t, neurons, runtype, batch_size):
    start = 2000
    inc = 1000

    eta = eta_start = 8.20

    for i in range(6):
        if t < start + i*inc:
            return eta/batch_size
        eta /= 2.0 # cut in half
    return eta/batch_size


def get_zeta_3(t, neurons, runtype, batch_size):
    start = 4000
    inc = 4000

    if t < start:
        return 1.25/batch_size
    if t < start + 1*inc:
        return 1.25/batch_size
    if t < start + 2*inc:
        return 0.6/batch_size
    if t < start + 3*inc:
        return 0.3/batch_size
    if t < start + 4*inc:
        return 0.07/batch_size
    return 0.03/batch_size

def get_zeta_1(t, neurons, runtype, batch_size):
    #start = 3000
    #inc = 2000
    start = 1000
    inc = 1000

    eta_start = 1.5

    if t < start:
        return (eta_start)/batch_size
    if t < start + 1*inc:
        return (eta_start/2)/batch_size
    if t < start + 2*inc:
        return (eta_start/4)/batch_size
    if t < start + 3*inc:
        return (eta_start/8)/batch_size
    if t < start + 4*inc:
        return (eta_start/16)/batch_size
    if t < start + 5*inc:
        return (eta_start/32)/batch_size
    if t < start + 6*inc:
        return (eta_start/64)/batch_size
    if t < start + 7*inc:
        return (eta_start/128)/batch_size
    if t < start + 8*inc:
        return (eta_start/256)/batch_size
    if t < start + 9*inc:
        return (eta_start/512)/batch_size
    if t < start + 10*inc:
        return (eta_start/1024)/batch_size
    return (eta_start/1024)/batch_size

import scipy.stats as stats

def my_dist(mean, point):
    sign = np.sign(mean - point)
    diff = np.abs(mean - point)

    max_dist = 20
    if diff > max_dist:
        return 0.0
    return sign * 1.0/(max_dist)

    if diff == 0:
        return 0.2
    elif diff == 1:
        return sign * 0.2
    elif diff == 2:
        return sign * 0.2
    elif diff == 3:
        return sign * 0.2
    elif diff == 5:
        return sign * 0.2
    return 0.0

def gauss(mean, point):
    return stats.norm(loc=mean,scale=1.00).pdf(point)

def initZ(neurons):
    if True:
        return np.eye(neurons)
    else:
        Z = np.zeros((neurons, neurons))
        for r in range(neurons):
            for c in range(neurons):
                Z[r,c] = my_dist(r, c)
        return Z
from math import sqrt

def initG(neurons, n, topographic):
    if topographic:
        if n == 2:
            G = np.zeros((neurons, neurons))
            for r in range(neurons):
                for c in range(neurons):
                    if c == r or c == r+1:
                        G[r,c] = 1
        elif n == 5:
            G = np.zeros((neurons, neurons))
            sqn = sqrt(neurons)
            for x in range(neurons):
                G[x,x] = 1
                if x % sqn != 0:
                    G[x,x-1] = 1
                if x >= sqn:
                    G[x, x - sqn] = 1
                if x % sqn != (sqn-1):
                    G[x,x+1] = 1
                if x < neurons - sqn:
                    G[x,x+sqn] = 1
        elif n == 9:
            G = np.zeros((neurons, neurons))
            sqn = sqrt(neurons)
            for x in range(neurons):
                U = x >= sqn
                L = x % sqn != 0
                R = x % sqn != (sqn-1)
                D = x < neurons - sqn

                if U:
                    if L:
                        G[x, x - sqn - 1] = 1
                    if R:
                        G[x, x - sqn + 1] = 1
                    G[x, x - sqn] = 1

                if L:
                    G[x,x-1] = 1
                if R:
                    G[x,x+1] = 1
                G[x,x] = 1

                if D:
                    if L:
                        G[x, x + sqn - 1] = 1
                    if R:
                        G[x, x + sqn + 1] = 1
                    G[x, x + sqn] = 1
        else:
            raise Exception("Unsupported topography")
    else:
        G = np.zeros((neurons, neurons))
        for x in range(neurons):
            start = (x/n) * n
            for i in range(start, start+n):
                G[x,i] = 1
    return G

#1 1
#1 1
    #1 1
    #1 1

  #a b c d e f g h i j k l m n o p
#a 1 1 - - 1 1 - - - - - - - - - -
#b 1 1 1 - 1 1 1 - - - - - - - - -
#c - 1 1 1 - 1 1 1 - - - - - - - -
#d - - 1 1 - - 1 1 - - - - - - - -
#e 1 1 - - 1 1 - - 1 1 - - - - - -
#f 1 1 1 - 1 1 1 - 1 1 1 - - - - -
#g - 1 1 1 - 1 1 1 - 1 1 1 - - - -
#h - - 1 1 - - 1 1 - - 1 1 - - - -
#i - - - - 1 1 - - 1 1 - - 1 1 - -
#j - - - - 1 1 1 - 1 1 1 - 1 1 1 -
#k - - - - - 1 1 1 - 1 1 1 - 1 1 1
#l - - - - - - 1 1 - - 1 1 - - 1 1
#m - - - - - - - - 1 1 - - 1 1 - -
#n - - - - - - - - 1 1 1 - 1 1 1 -
#o - - - - - - - - - 1 1 1 - 1 1 1
#p - - - - - - - - - - 1 1 - - 1 1
#

  # 4 neighbors
  #a b c d e f g h i j k l m n o p
#a 1 1 - - 1 - - - - - - - - - - -
#b 1 1 1 - - 1 - - - - - - - - - -
#c - 1 1 1 - - 1 - - - - - - - - -
#d - - 1 1 - - - 1 - - - - - - - -
#e 1 - - - 1 1 - - 1 - - - - - - -
#f - 1 - - 1 1 1 - - 1 - - - - - -
#g - 1 1 1 - 1 1 1 - - 1 - - - - -
#h - - 1 1 - - 1 1 - - - 1 - - - -
#i - - - - 1 1 - - 1 1 - - 1 - - -
#j - - - - 1 1 1 - 1 1 1 - - 1 - -
#k - - - - - 1 1 1 - 1 1 1 - - 1 -
#l - - - - - - 1 1 - - 1 1 - - - 1
#m - - - - - - - - 1 1 - - 1 1 - -
#n - - - - - - - - 1 1 1 - 1 1 1 -
#o - - - - - - - - - 1 1 1 - 1 1 1
#p - - - - - - - - - - 1 1 - - 1 1
#

 #[ 1 1 - - 1 - - - - - - - - - - -]
 #[ 1 1 1 - - 1 - - - - - - - - - -]
 #[ - 1 1 1 - - 1 - - - - - - - - -]
 #[ - - 1 1 - - - 1 - - - - - - - -]
 #[ 1 - - - 1 1 - - 1 - - - - - - -]
 #[ - 1 - - 1 1 1 - - 1 - - - - - -]
 #[ - - 1 - - 1 1 1 - - 1 - - - - -]
 #[ - - - 1 - - 1 1 - - - 1 - - - -]
 #[ - - - - 1 - - - 1 1 - - 1 - - -]
 #[ - - - - - 1 - - 1 1 1 - - 1 - -]
 #[ - - - - - - 1 - - 1 1 1 - - 1 -]
 #[ - - - - - - - 1 - - 1 1 - - - 1]
 #[ - - - - - - - - 1 - - - 1 1 - -]
 #[ - - - - - - - - - 1 - - 1 1 1 -]
 #[ - - - - - - - - - - 1 - - 1 1 1]
 #[ - - - - - - - - - - - 1 - - 1 1]

#a b c d
#e f g h
#i j k l
#m n o p
