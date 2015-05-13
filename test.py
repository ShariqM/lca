import matplotlib.pyplot as plt

#classes = ['A','A','B','C','C','C']
#colours = ['r','r','b','g','g','g']
#for (i,cla) in enumerate(set(classes)):
    #xc = [p for (j,p) in enumerate(x) if classes[j]==cla]
    #yc = [p for (j,p) in enumerate(y) if classes[j]==cla]
    #cols = [c for (j,c) in enumerate(colours) if classes[j]==cla]
    #plt.scatter(xc,yc,c=cols,label=cla)
#plt.legend(loc=4)


import pdb
import numpy as np

p = 144
t = 10
n = 288

R = np.random.random((p)) # p is really n but this is just a check
cR = np.tile(R, (n, 1)).T
Gam = np.random.random((p,n,t))

c = np.tensordot(cR, Gam, 2)

cloop = np.zeros(t)
for i in range(t):
    for j in range(p):
        for k in range(n):
            cloop[i] += R[j] * Gam[j,k,i]

assert np.allclose(c, cloop)



u = np.random.random((p))
g = np.tensordot(Gam, c, 1)
r = np.dot(g.T, u)


b = 100
u = np.random.random((n,b))
c = np.random.random((t,b))

pdb.set_trace()
x = np.tensordot(Gam, c, 1)
r = np.einsum('pnb,nb->pb', x, u)
#r = np.tensordot(x, u, [[1,], [0,]])
assert r.shape == (p,b)
