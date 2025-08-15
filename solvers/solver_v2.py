import numpy as np
import scipy.integrate as sc

def integrate(ys, xs):
    """
    Integrator using Scipy's Simpson's rule
    """
    return sc.simpson(ys,xs)


def euler_method(f, h, rhs_func):
    """
    Euler's method
    """    
    f_new = f + h * rhs_func

    return f_new


# Functions for estimating the Boltzmann integrals
def eps(x,y):
    """
    Dimensionless energy (Energy/Temperature)
    """
    return np.sqrt(x*x + y*y)


def F1(x,y1, y2, f_y1, f_y2):
    
    return -f_y1 * f_y2 + np.exp(-eps(x,y1) - eps(x,y2))


def F2(x,y1, y2, f_y1, f_y2):
    
    return -f_y1 + f_y2 * np.exp(-eps(x,y1)-eps(x,y2))

def I(x, y1, y2, f_y1, f_y2):
    G = 1 
    return G * G / 2 * F1(x, y1, y2, f_y1, f_y2)

def H(x, y1, y3, f_y1, f_y3):
    G = 1
    return G * G * 2 /(y1 * y3) * (np.exp(- abs(y1 - y3)/2) - np.exp(-(y1 + y3)/2)) * F2(x, y1, y3, f_y1, f_y3)


def C_ann(x:float, y1:float, y2s:list, f_y1:float, f_y2s:list) -> float:
    """
    Dimensionless Annihilation Operator - integrand
    """
    # define the integrand
    integrand = lambda x, y1, y2, f_y1, f_y2: y2 * y2 /(4 * eps(x,y1) * eps(x, y2)) * I(x, y1, y2, f_y1, f_y2) * F1(x, y1, y2, f_y1, f_y2)

    # compute the integrand for different y3-values
    Fs = [integrand(x,y1,y2,f_y1,f_y2) for y2, f_y2 in zip(y2s, f_y2s)]


    return 1 / (2*np.pi)**3 * integrate(Fs,y2s)


def C_scatter(x:float, y1:float, y3s:list,f_y1:float,f_y3s:list) -> float:
    """
    Dimensionless Scattering Operator
    """
    # define the integrand
    integrand = lambda x, y1, y3, f_y1, f_y3: y3 * y3 / (4 * eps(x, y1) * eps(x, y3)) * H(x, y1, y3, f_y1, f_y3) * F2(x, y1, y3, f_y1, f_y3)

    # compute the integrand for different y3-values
    Fs = [integrand(x,y1,y3,f_y1,f_y3) for y3, f_y3 in zip(y3s, f_y3s)]    

    return 1 / (2 * np.pi)**3 * integrate(Fs,y3s)


def BE_solver(xmin:float, xmax:float,ys:list,initial_condition,collisions, dx = 1e-2) -> list:
    """
    Function that solves the BE for the distribution function given
    the starting and ending point of (dimensionless) time and the
    collision operators for the interaction and a list of momentum
    values. 


    xmin                -> float                :   Takes the smallest x-value 
    xmax                -> float                :   Takes the largest x-value
    ys                  -> list                 :   List of all dimensionless momentum values
    initial_condition   -> function (-> float)  :   initial values that computes f at x0
    collisions          -> function (-> float)  :   Computes the collision integrals

    """

    fs =  [[initial_condition(y,xmin)] for y in ys]             # Initial distribution function matrix
    xs = []                                                     # Time steps

    N = len(ys) # # momentum values

    x = xmin    # Initial value of x
    i = 0       # iteration

    while (x < xmax):
        for j in range(N):

           
            # Compute the derivative w.r.t. y      
            if j == 0:
                df_dy = (fs[j+1][i] - fs[j][i]) / (ys[j+1] - ys[j])
            elif j == len(ys) - 1:
                df_dy = (fs[j][i] - fs[j-1][i]) / (ys[j] - ys[j-1])
            else:
                df_dy = (fs[j+1][i] - fs[j-1][i]) / (ys[j+1] - ys[j-1])

            f_ys = [fs[k][i] for k in range(N)]

            # Compute the right-hand side
            y = ys[j]
            
            
            rhs_func = 1/x * (y * df_dy + collisions(x,ys[j],ys,fs[j][i],f_ys))
                
            f_euler = euler_method(fs[j][i],dx,rhs_func)                      

            fs[j].append(f_euler)
        xs.append(x)
        x += dx
        i += 1
    xs.append(x)


    return (xs,fs)