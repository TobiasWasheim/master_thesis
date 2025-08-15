"""

Algorithm for solving the Boltzmann equation

"""
import numpy as np
import scipy.linalg as lin



def BE_solver(xmin: float, xmax: float,ys: list,initial_condition):
    fs = [initial_condition(y,xmin) for y in ys]     # define the grid of distribution function values with initial values
    xs = [xmin]                                      # list of x (time) values

    tol = 1e-6      # tolerance
    saf = 1         # safety
    err = 0         # error


    x = xmin
    i = 0       # iterations

    while (x < xmax):
        
        local_err = []          # define local error array  
        alpha = lambda x: 0     # define the front factor that are infront of the collision operators


        for k in range(len(ys)):        # looping over all momenta

            # compute the collision integrals
            collision_ann = 0
            collision_sca = 0

            # define the k-factors
            k1 = 0
            k2 = 0  

            # define distribution function step forward
            f_euler = 0
            f_heun = 0

            local_err.append(f_heun - f_euler)
            

        err = lin.norm(local_err)                           # compute the error as the norm of the local errors
        if err < 1:
            dx = min(saf * dx * np.sqrt(tol/abs(err)),2 * dx)   # adapt x (time) step-size
            dx = max(dx, 1e-12)                                 # make sure that the smallest time step x >= 1e-12
                    
        x += dx
        xs.append(x)
        i += 1
    
    return (xs, ys, fs)
