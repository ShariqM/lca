import matplotlib.pyplot as plt
import numpy as np
from numpy.random import rand
import pdb

#activity_log = np.load('coeff_Bruno__initP30_100t.npy')
#activity_log = np.load('coeff_Bruno__init080_100t.npy')
activity_log = np.load('coeff_S193__initP20_100t.npy')
pdb.set_trace()


x_num = 657
y_num = 668
log_x = np.zeros(0)
log_y = np.zeros(0)
batch_size = 256

tstart = 75
tend = 99
colors = np.linspace(0, 1, tend - tstart)

scats = []
idx = 1

found=1
corr_check=True

y_num = 935
for i in range(batch_size):
    # Correlation Check
    if i != 241:
        continue

    if corr_check:
        x_vec = activity_log[x_num, i, :]
        for y in range(activity_log.shape[0]):
            if y == x_num:
                continue
            y_vec = activity_log[y, i, :]
            corr = np.dot(x_vec, y_vec.T)
            if np.abs(corr) > 0.4:
                print "Y=%d P=%d Corr=%f" % (y, i, corr)

    log_x = activity_log[x_num, i, :][tstart:tend]
    log_y = activity_log[y_num, i, :][tstart:tend]
    print 'Points: ', len(log_x)

    if found:
        #if max(log_y) < 0.40:
            #continue
        #if min(log_y) > -1.0:
            #continue
        cmap = plt.get_cmap(['autumn','cool'][idx % 2])
        idx += 1
        scats.append(plt.scatter(log_x, log_y, s=8, c=colors, label='Patch %d' % i, cmap=cmap, lw=0))
    else:
        color = (rand(), rand(), rand())
        plt.scatter(log_x, log_y, s=8, c=color, label='Patch %d' % i, lw=0)

mag = 0.5
plt.plot((-mag,0,0,0,0,mag), (0,0,-mag,mag,0,0), 'b') # Axis

if found:
    for scat in scats:
        cbar = plt.colorbar(scat, ticks=[0, 0.5, 1])
        cbar.ax.set_yticklabels(['t=%d' % tstart, 't=%d' % np.average((tstart, tend)), 't=%d' % tend])

plt.xlabel("%d Activity" % x_num, fontdict={'fontsize':18})
plt.ylabel("%d Activity" % y_num, fontdict={'fontsize':18})
plt.axis('equal')
plt.legend(loc=4, prop={'size':9})
plt.show(block=True)
