#!/usr/bin/env python3

import numpy as np


EPSILON = 0.0001 # error parameter of Solver


def rk4_solver(func, t0, t1, y0, h=EPSILON):

    ''' Solve y' = func(t, y) system with Runge-Kata-4 mehod. y can be array for
        system of equations.
    '''

    y0 = np.array(y0)
    N = int((t1-t0) / h)  # Number of steps
    dim = y0.shape[0] # dimension of y
    t_interval = np.linspace(t0, t1, N)
    y_matrix = np.empty((N,dim), dtype = "float")
    y_matrix[0] = y0    # A Matrix where y answers storage
    y = y0

    for i in range(N-1) :
        k1 = func(t_interval[i], y)
        k2 = func(t_interval[i]+h/2, y+h*k1/2)
        k3 = func(t_interval[i]+h/2, y+h*k2/2)
        k4 = func(t_interval[i]+h, y+h*k3)
        y = y+1/6*h*(k1 + 2*k2 + 2*k3 + k4) # This is Runge-Kata-4 method
        y_matrix[i+1] = y

    return t_interval, y_matrix

''' Here is an usage example

from RK4 import rk4_solver
import numpy as np
import matplotlib.pyplot as plt


def function(t, y):
    y1, y2 = y
    return np.array([y2, -y1])


t, y = rk4_solver(function, 0.0, 20.0, [1, 0])
plt.plot(t, y)
plt.show()

'''
