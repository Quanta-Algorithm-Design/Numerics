#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt


EPSILON = 0.001 # error parameter of Solver


def rk4_solver(func, t0, y0, t1, h = EPSILON):

    ''' Solve y' = func(t, y) system with Runge-Kata-4 mehod. y can be array for
        system of equations.
    '''

    N = int((t1-t0)/h)
    t = np.linspace(t0, t1, N)
    y = []
    yn = y0
    for tn in t :
        k1 = func(tn, yn)
        k2 = func(tn+h/2, yn+h*k1/2)
        k3 = func(tn+h/2, yn+h*k2/2)
        k4 = func(tn+h, yn+h*k3)
        yn = yn+1/6*h*(k1+2*k2+2*k3+k4)
        y.append(yn)
    y = np.array(y)
    return t, y


def main (): # Tests a simple example
    def function(t, y):
        y1, y2 = y
        return np.array([y2, -y1])


    t, y = rk4_solver(function, 0.0, [1, 0], 10.0)
    plt.plot(t, y)
    plt.show()


if __name__ == '__main__':
    main()
