import theano.tensor as T
from theano import *
x = T.dscalar('x')
y = x ** 2
gy = T.grad(y, x)
f = function([x], gy)
print f(4)
print f(94.2)
