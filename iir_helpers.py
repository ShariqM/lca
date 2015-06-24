import socket
import pdb
import numpy as np
from theano import *
import theano.tensor as T
from theano.tensor import tensordot as ttdot

#Av = T.tensor3('A')
#Bv = T.fmatrix('B')
#Cv = T.tensor3('C')
#Dv = T.fmatrix('D')
#x = ttdot(Av, Bv, [[0], [0]])
#y = ttdot(Cv, Dv, [[1], [0]])
#o = ttdot(x, y, [[1,2], [1,0]]).T
#
#grad_a_TT = theano.function([Av, Bv, Cv, Dv], o, allow_input_downcast=True)

Av = T.fmatrix('A')
Bv = T.fmatrix('B')
o2 = T.dot(Av, Bv)
t2dot = theano.function([Av, Bv], o2, allow_input_downcast=True)

Cv = T.fmatrix('C')
Dv = T.tensor3('D')
o3 = T.tensordot(Cv, Dv, [[1], [0]])
treconstruct = theano.function([Cv, Dv], o3, allow_input_downcast=True)


Ev = T.tensor3('E')
Fv = T.tensor3('F')
#o3 = T.batched_dot(Ev.dimshuffle(2, 1, 0), Fv.dimshuffle(2, 1, 0))
#o3 = T.batched_dot(Ev, Fv.dimshuffle(2, 0, 1)).dimshuffle(1,2,0)
#o3 = T.batched_dot(Ev, Fv)
o3 = T.batched_dot(Ev, Fv)
t_bdot = theano.function([Ev, Fv], o3, allow_input_downcast=True)
