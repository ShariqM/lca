import socket
import pdb
import numpy as np
from theano import *
import theano.tensor as T
ttdot = T.tensordot

Av = T.tensor3('A')
Bv = T.tensor3('B')
o = T.tensordot(Av, Bv, [[1,2], [0,2]])
tendot = theano.function([Av, Bv], o, allow_input_downcast=True)

Cv = T.tensor3('C')
Dv = T.tensor3('D')
o2 = T.tensordot(Cv, Dv, [[0,2], [0,2]])
ten2dot = theano.function([Cv, Dv], o2, allow_input_downcast=True)

Ev = T.tensor3('E')
Fv = T.tensor3('F')
Gv = T.fmatrix('G')
o3 = ttdot(ttdot(Gv, Ev, [[0], [0]]), Fv, [[0,2], [0,2]]).T
ten3dot = theano.function([Ev, Fv, Gv], o3, allow_input_downcast=True)

Hv = T.fmatrix('H')
Iv = T.tensor3('I')
Jv = T.tensor3('J')
o4 = ttdot(Hv, ttdot(Iv, Jv, [[1,2], [0,2]]), [[1], [0]])
ten3dot2 = theano.function([Hv, Iv, Jv], o4, allow_input_downcast=True)

Hv = T.fmatrix()
Iv = T.fmatrix()
Jv = T.fmatrix()
o4 = ttdot(Hv, ttdot(Iv, Jv, [[0], [0]]), [[1], [1]])
ten_2_2_2 = theano.function([Hv, Iv, Jv], o4, allow_input_downcast=True)
