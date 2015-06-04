import socket
import pdb
import numpy as np
from theano import *
import theano.tensor as T

Av = T.tensor3('A')
Bv = T.tensor3('B')
o = T.tensordot(Av, Bv, [[1,2], [0,2]])
tendot = theano.function([Av, Bv], o, allow_input_downcast=True)
