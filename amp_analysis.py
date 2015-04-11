import pdb
import scipy.io
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from showbfs import showbfs

Phi = scipy.io.loadmat('dict/Phi_194_1.2.mat')['Phi']
#Phi = scipy.io.loadmat('dict/Phi_193_37.0.mat')['Phi']
#Phi = scipy.io.loadmat('dict/Phi_169_45.0.mat')['Phi'] # 117 & 132
#Phi = scipy.io.loadmat('dict/Bruno_FIELD_Phi256x1024.mat')['Phi'] # 631 and 642
showbfs(Phi)
plt.show()
show = False

prod = np.dot(Phi.T, Phi)
#np.fill_diagonal(prod, 20.01) # No identitical indexes
#rprod = prod.ravel()
#plt.hist(rprod)
#plt.show()
indexes = np.where(prod < 0.5 * np.min(prod))
print 'Found %d low candidates' % len(indexes[0])

aPhi = np.abs(Phi)
aprod = np.dot(aPhi.T, aPhi)
np.fill_diagonal(aprod, 0.01) # No identitical indexes
#raprod = aprod.ravel()
if show:
    plt.hist(raprod)
    plt.show()

print 'Max aprod %f, Min prod %f' % (np.max(aprod), np.min(prod))
aindexes = np.where(0.5 * np.max(aprod) < aprod)
print 'Found %d high candidates' % len(aindexes[0])
# (array([ 0, 11, 12, 23]), array([12, 23,  0, 11]))
# 120 and 132 seem good
# 239 1000 0.849423
# 29 434 0.914251
# 163 600 0.909806
# 198 76 0.914342
# 198 415 0.903784
# 214 353 0.90408

# Matchers
# 76 198 0.914342
# 88 1003 0.924501
# 128 666 0.919582
# 206 854 0.925844

#print np.max(aprod)

indexes = set(zip(*indexes)).intersection(set(zip(*aindexes)))
#indexes = zip(*aindexes)
print 'Found %d combined candidates' % len(indexes)

pdb.set_trace()
for i in range(Phi.shape[1]):
    j = i + Phi.shape[1]/2

#for (i,j) in indexes:
#for (i,j) in ((403, 703),):
#for (i,j) in ((868, 919),):
#for (i,j) in ((29, 434),):
#for (i,j) in ((117, 132),):
#for (i,j) in ((59, 98),): # Bruno
    if i > j:
        continue # Don't look both dirs
    print i, j, aprod[i][j], np.dot(Phi[:,i], Phi[:,j])
    pair = np.concatenate((np.array([Phi[:,i]]), np.array([Phi[:,j]])))
    showbfs(pair)
    plt.title("I=%d, J=%d Corr=%f" % (i, j, aprod[i][j]))
    plt.show()
    #pair = np.concatenate((np.array([aPhi[:,i]]), np.array([aPhi[:,j]]))).T
    #plt.show()

# Candidates
# 29 434 0.914251
# 76 198 0.914342
# 146 277 0.923386
# 193 822 0.923862
# 198 415 0.903784
# 214 353 0.90408
# 313 362 0.92021
# 316 461 0.900871
# 403 703 0.903446 *
# 419 837 0.91666
# 553 681 0.91558
# 868 919 0.909754 *
# 904 968 0.900097 *


# Shariq 193
# 42 294
# 657 668 *
# 105 623 **


# Bruno 256x1024
# 17 1023 0.811147689831
# 19 637 0.847688420821 *
# 19 656 0.811423362773
# 23 119 0.823991306893
# 56 179 0.823908583262 *?
# 57 364 0.856591791215
# 59 98 0.801490076113 **
# 59 154 0.804692579443 *
# 98 154 0.807563182685
# 116 676 0.825526092553
# 151 408 0.818621641307 *
# 342 888 0.566662136281 -0.435155257836
