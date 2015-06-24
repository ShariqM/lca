import socket
import pdb
import numpy as np
from theano import *
import theano.tensor as T
from theano.tensor import tensordot as ttdot

Av = T.fmatrix('A')
Bv = T.fmatrix('B')
o2 = T.dot(Av, Bv)
t2dot = theano.function([Av, Bv], o2, allow_input_downcast=True)

Cv = T.tensor3('C')
Dv = T.tensor3('D')
o3 = T.batched_dot(Cv, Dv)
t_bdot = theano.function([Cv, Dv], o3, allow_input_downcast=True)

Ev = T.fmatrix('E')
Fv = T.fmatrix('F')
o4 = T.tensordot(Ev, Fv, [[1], [0]])
t_tdot = theano.function([Ev, Fv], o4, allow_input_downcast=True)


Ev = T.fmatrix('E')
Fv = T.fmatrix('F')
Gv = T.fmatrix('G')
o5 = T.dot(Ev, T.dot(Fv, Gv))
t3dot = theano.function([Ev, Fv, Gv], o5, allow_input_downcast=True)


