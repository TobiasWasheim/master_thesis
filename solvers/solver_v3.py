"""

Algorithm for solving the Boltzmann equation - using scipy.integrate's solve_ivp

"""
import numpy as np
import scipy.linalg as lin
import scipy.integrate as sc

def H(x):
    return 0


def BE_solver(xmin: float, xmax: float,ys: list,initial_condition):
    fs = [initial_condition(y,xmin) for y in ys]     # define the grid of distribution function values with initial values
    xs = [xmin]                                      # list of x (time) values

    x = xmin

    while (x < xmax):

        # define a right-hand side of the Boltzmann equation
        rhs = df_dy + 1/( x * H(x)) * collisions()

        sc.solve_ivp(rhs, t_span, y0, method="Radau",t_eval = None, dense_output=False, events = None, vectorized=None, args = None)


    return (xs, ys, fs)
