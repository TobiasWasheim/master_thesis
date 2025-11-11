from numba import njit
import numpy as np
import scipy.special as sc

############################

# Initial value functions

############################

def density_equilibrium(x):
    """
    Initial number density function at full thermal equilibrium
    """
    gi = 2
    gs = 1
    solution = gi * 45 / (4 * np.pi**4) * x*x / gs * sc.kn(2,x)
    return solution

def equilibrium(y,x):
    """
    Full thermal equilibrium
    """
    return np.exp(-np.sqrt(x*x+y*y)) # full thermal equilibrium = e^(-E/T)

def random_distribution(y,x):    
    """
    Random initial distribution
    """
    return np.random.random() # generates a random float between 0.0 and 1.0

@njit
def E(y,x):
    """
    Energy: E = sqrt(x^2+y^2)
    """
    return np.sqrt(y*y+x*x)


############################

# Functions to test the integrator

############################

def f0(y,x):
    return 1 + y

def test_solution(y,x):
    return np.exp(-x) * (1 + y) + np.sin(np.pi * y) * 1.5 / (2/np.pi) * (np.exp((2/np.pi - 1)*x) - np.exp(-x))

@njit
def test_function(f1,f3,y1,y3,x,Gamma):
    """
    Test case for the numerical Boltzmann Equation Solver.
    """
    return - f1 + np.sin(np.pi * y1) * f3


############################

# integrands of the 1 dimensional collision operators

############################


@njit
def scattering(f1,f3,y1,y3,x,Gamma):
    """
    Elastic scattering collision operator for cold dark matter interacting with light-weight standard model
    particles
    """
    J = y3 / E(y3,x) * (y1 + y3 - abs(y1 - y3))/y1
    Lambda = -f1 + f3 * np.exp(-(E(y1,x) - E(y3 ,x)))
    front_factor = 1/(E(y1,x))
    return Gamma * front_factor * J * Lambda

@njit
def s_annihilation(f1,f2,y1,y2,x,Gamma):
    """
    S-wave annihilation collision operator for cold dark matter interacting with light-weight standardard-model 
    particles
    """
    front_factor = 1/E(y1,x)
    J = y2 * y2 / E(y2,x)
    Lambda = -f1 * f2 + np.exp(-E(y1,x)-E(y2,x))
    return Gamma * front_factor * J * Lambda

@njit
def p_annihilation(f1,f2,y1,y2,x,Gamma):
    """
    P-wave annihilation collision operator for cold dark matter interacting with light-weight standardard-model 
    particles
    """
    J = y2 * y2 / E(y2,x) * (y1 * y1 + y2 * y2) / (x * x)
    Lambda = -f1 * f2 + np.exp(-E(y1,x)-E(y2,x))
    return Gamma * 1/E(y1,x) * J * Lambda

@njit
def annihilation_scattering(f1,f2,y1,y2,x,Gamma):
    """
    Annihilation and scattering
    """
    return s_annihilation(f1,f2,y1,y2,x,Gamma) + scattering(f1,f2,y1,y2,x,Gamma)

@njit
def scattering_units(f1,f3,p1,p3,t,Gamma):
    m = 1
    E1 = E(p1,t)
    E3 = E(p3,t)

    Lambda = -f1 + f3 * np.exp(-(E1-E3)*np.sqrt(t))
    J = p3 / E3 * (p1+p3 - abs(p1-p3))/p1
    front_factor = 1/np.sqrt(t) * 1/E1
    return Gamma * front_factor * J * Lambda