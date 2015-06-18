import socket
import pdb
import numpy as np
from theano import *
import theano.tensor as T
from theano.tensor import tensordot as ttdot

Av = T.tensor3('A')
Bv = T.fmatrix('B')
Cv = T.tensor3('C')
Dv = T.fmatrix('D')
x = ttdot(Av, Bv, [[0], [0]])
y = ttdot(Cv, Dv, [[1], [0]])
o = ttdot(x, y, [[1,2], [1,0]]).T

grad_a_TT = theano.function([Av, Bv, Cv, Dv], o, allow_input_downcast=True)
