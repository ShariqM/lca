import pdb

from math import log
from math import sin, cos
from math import exp
from math import isnan
import random
import scipy.io
import numpy as np
from datetime import datetime
import time

import matplotlib.pyplot as plt
from matplotlib import cm

import sys
import os

class Graphs():
    def __init__(self):
        pass

    def power_vs_exp(self):
        x = np.arange(0.1,100, 0.1)
        exp = np.exp(-x)
        power = 1./x

        plt.subplot(221)
        plt.axis('equal')
        plt.plot(x[:100], exp[:100], label='exp')
        plt.plot(x[:100], power[:100], label='power')
        plt.xlabel('X', fontdict={'fontsize':12})
        plt.ylabel('Y', fontdict={'fontsize':12})
        plt.legend()

        plt.subplot(222)
        plt.axis('equal')
        plt.plot(np.log(x), exp, label='exp')
        plt.plot(np.log(x), power, label='power')
        plt.xlabel('log(X)', fontdict={'fontsize':12})
        plt.ylabel('Y', fontdict={'fontsize':12})
        plt.legend()

        plt.subplot(223)
        plt.axis('equal')
        plt.plot(x, np.log(exp), label='exp')
        plt.plot(x, np.log(power), label='power')
        plt.xlabel('X', fontdict={'fontsize':12})
        plt.ylabel('log(Y)', fontdict={'fontsize':12})
        plt.legend()

        plt.subplot(224)
        plt.axis('equal')
        plt.plot(np.log(x), np.log(exp), label='exp')
        plt.plot(np.log(x), np.log(power), label='power')
        plt.xlabel('log(X)', fontdict={'fontsize':12})
        plt.ylabel('log(Y)', fontdict={'fontsize':12})
        plt.legend()
        plt.show()

    def log_1plusx2(self):
        x = np.arange(-8,8.01, 0.1)
        log_1plusx2 = np.log(1 + x ** 2)

        plt.plot(x, log_1plusx2, label='log(1+x^2)')
        plt.plot(x, np.abs(x), label='abs(x)')
        plt.plot(x, 2 * np.abs(x), label='2 * abs(x)')
        plt.plot(x, np.sqrt(np.abs(x)), label='sqrt(abs(x))')
        #plt.plot(x, 1./(.01+np.abs(x)), label='1/abs(x)')
        plt.title("Cost Functions")
        plt.xlabel('X', fontdict={'fontsize':12})
        plt.ylabel('Y', fontdict={'fontsize':12})
        plt.legend()
        plt.show()

    def first_order(self):
        dt = 0.05
        x = np.array([1.0, -0.0]).T

        A = np.array([[0, -1], [1, 0]])
        A = np.array([[1.1, -1], [1, 1.1]])
        R = np.array([[cos(dt), -sin(dt)], [sin(dt), cos(dt)]])

        steps = 200
        scats = []
        cm = plt.cm.get_cmap('autumn')

        data_x1 = []
        data_x2 = []
        for i in range(steps):
            x = dt * np.dot(A, x) + (1 - dt) * x
            #x = np.dot(R, x)
            print '%f, %f' % (x[0], x[1])
            data_x1.append(x[0])
            data_x2.append(x[1])

        sc = plt.scatter(data_x1, data_x2, c=range(steps), vmin=0, vmax=steps, s=8, cmap=cm)
        plt.colorbar(sc)
        mag = 0.5
        plt.plot((-mag,0,0,0,0,mag), (0,0,-mag,mag,0,0), 'b') # Axis
        plt.axis('equal')
        plt.show()

g = Graphs()
#g.power_vs_exp()
#g.log_1plusx2()
g.first_order()
