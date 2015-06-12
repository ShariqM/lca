import math
from math import *
from numpy.fft import *
import numpy as np
import matplotlib.pyplot as plt
import pdb

# Sample points
t = 150.0
it = int(t)
nn = int(t * 1)

# Sample rate
dt = 1/t
x = range(0,150)
loglog = False

ploti = 1
for a in (0.9, 0.8, 0.7):
    #y = (1/3.333) * np.power(a, x)
    y = [0] * it
    y[(it/4):(3*it/4)] = [1] * (it/2)
    print y
    pdb.set_trace()

    # Plot in time domain
    plt.subplot(int("23%d" % ploti))
    plt.plot(y)

    yf = fft(y)
    yplot = fftshift(yf); # Center at 0
    xf = fftshift(fftfreq(int(nn),dt)); # x axis points

    my = max(yplot)
    print 'my ', my
    #yplot /= my

    print 'For a=%f' % a
    #for f in (0, 45, 75):
        #print '\ty%d: %.3f + %.3fi e-2: %.3f' % (f, yplot[f].real, yplot[f].imag, exp(-2))
    #print ''

    # Plot DFT
    plt.subplot(int("23%d" % (3+ploti)))
    plt.title("a=%.3f" % a)

    if loglog:
        yplot = yplot[75:]
        xf = xf[75:]
        plt.xscale('log')
        plt.yscale('log')
    else:
        plt.plot(xf, [0] * len(xf), color='k', label='X axis')

    plt.plot(xf, np.real(yplot), color='b', label='real')
    plt.plot(xf, np.imag(yplot), color='g', linestyle='--', label='imag')
    plt.legend()

    ploti += 1
plt.show()
