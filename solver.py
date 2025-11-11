import numpy as np
import sys
import scipy.integrate as sc
from numba import njit
from collisions import E

def solver(x_span,y_span,Gamma,initial_condition, collision, rtol=1e-6,atol=1e-6) -> tuple:
    """
    Solves coupled integro ordinary differential equation 
    x_span = range for x-values
    y_span = range for y-values and number of grid points
    Gamma = interaction strength
    """
    ys = ys = np.linspace(y_span[0], y_span[1], y_span[2])
    f0 = np.array([initial_condition(y, x_span[0]) for y in ys], dtype=np.float64)
      
    def RHS_wrapper(x,fs):
        return collision_operator(x,fs,ys,Gamma,collision)
    
    # Solve the Integro-ODE using Scipy's ODE solver from x0,...,xN
    sol = sc.solve_ivp(fun=RHS_wrapper, t_span=x_span, y0=f0, method="BDF", rtol=rtol,atol=atol)
    xs = sol.t
    fs = sol.y      

    return (xs, ys, fs)

@njit
def trapezoid(y:list, x:list) -> float:
    return np.trapezoid(y,x)

@njit
def collision_operator(x:float, fs:list,ys:list,Gamma:float,collision) -> list:
    """
    Computes a vector of collision operators for different y1s
    """
    N = len(fs)
    rhs = np.empty(N)
    for i in range(N):
        f1 = fs[i]
        y1 = ys[i]
        integrand_values = np.empty(N)
        for j in range(N):
            f3 = fs[j]
            y3 = ys[j]
            integrand_values[j] = collision(f1, f3, y1, y3, x,Gamma)
        rhs[i] = trapezoid(integrand_values, ys)
    return rhs

def energy_conservation(x_span,y_span,Gamma,initial_condition,collision,rtol=1e-6,atol=1e-6):

    solv = solver(x_span,y_span,Gamma,initial_condition,collision,rtol,atol)

    xs = solv[0]
    ys = solv[1]
    fs = solv[2]

    # computing n_dot and E_dot for scattering
    N = len(fs)
    n = len(xs)
    n_dot = np.empty(n)
    E_dot = np.empty(n)
    for k in range(n):   
        n_dot_integrand = np.empty(N)    
        E_dot_integrand = np.empty(N) 
        for i in range(N):
            f1 = fs[i][k]
            y1 = ys[i]
            integrand_values = np.empty(N)
            for j in range(N):
                f3 = fs[j][k]
                y3 = ys[j]
                integrand_values[j] = collision(f1, f3, y1, y3, xs[k],Gamma)
            collision_op = trapezoid(integrand_values, ys)
            n_dot_integrand[i] = collision_op * ys[i] * ys[i]
            E_dot_integrand[i] = n_dot_integrand[i] * E(y1, xs[k])
        n_dot[k] = trapezoid(n_dot_integrand,ys) 
        E_dot[k] = trapezoid(E_dot_integrand,ys)
    
    return (n_dot,E_dot,xs)


def number_density(fs,ys) -> list:
    gi = 2
    ns = np.empty(len(fs[0]))
    for j in range(len(fs[0])):
        yyfs = [ys[i] * ys[i] * fs[i][j] for i in range(len(ys))]
        ns[j] = gi * 1/(np.pi)**3 * trapezoid(yyfs,ys)
    return ns