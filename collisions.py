from numba import njit
import numpy as np

@njit
def test_function(f1,f3,y1,y3,x):
    """
    Test case for the numerical Boltzmann Equation Solver.
    """
    return - f1 + np.sin(np.pi * y1) * f3

@njit
def E(y,x):
    """
    Energy - E = sqrt(x^2+y^2)
    """
    return np.sqrt(y*y+x*x)

def equilibrium(y,x):
    """
    Full thermal equilibrium - f = exp(-E)
    """
    return np.exp(-E(y,x))

@njit
def scattering(f1,f3,y1,y3,x):
    """
    Elastic scattering collision operator for cold dark matter interacting with light-weight standard model
    particles
    """
    J = y3 / E(y3,x) * (y1 + y3 - abs(y1 - y3))/y1
    Lambda = -f1 + f3 * np.exp(-(E(y1,x) - E(y3,x)))

    return 1/E(y1,x) * J * Lambda

@njit
def s_annihilation(f1,f2,y1,y2,x):
    """
    S-wave annihilation collision operator for cold dark matter interacting with light-weight standardard-model 
    particles
    """
    front_factor = 1/E(y1,x)
    J = y2 * y2 / E(y2,x)
    Lambda = -f1 * f2 + np.exp(-E(y1,x)-E(y2,x))
    return front_factor * J * Lambda

@njit
def p_annihilation(f1,f2,y1,y2,x):
    """
    P-wave annihilation collision operator for cold dark matter interacting with light-weight standardard-model 
    particles
    """
    J = y2 * y2 / E(y2,x) * (y1 * y1 + y2 * y2) / (x * x)
    Lambda = -f1 * f2 + np.exp(-E(y1,x)-E(y2,x))
    return 1/E(y1,x) * J * Lambda

@njit
def annihilation_scattering(f1,f2,y1,y2,x):
    return s_annihilation(f1,f2,y1,y2,x) + scattering(f1,f2,y1,y2,x)