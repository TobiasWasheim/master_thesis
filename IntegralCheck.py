import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

def Lambda(t,p1,p3):
    return 1/p3

def collision_1D(t,p1,p3,Lambda):
    
    S = 0.5
    T = 5
    E1 = np.sqrt(t*t + p1*p1)
    E3 = np.sqrt(t*t + p3*p3)

    front_factor = S * T /(2*np.pi)**3 * 1 / E1

    integrand = p3 /  E3 * 0.5 * (p1+p3 - abs(p1-p3))/p1
    
    return T * integrand * Lambda(t,p1,p3)

def collision_2D(t,p1,p2,p3,Lambda):
    
    S = 0.5
    T = 5
    E1 = np.sqrt(t*t + p1*p1)
    E3 = np.sqrt(t*t + p3*p3)

    front_factor = S  /(2*np.pi)**3 * 1 / E1

    integrand = p3 / (2 * E3) * 0.5 * (p1 + p3 - abs(p1-p3))/p1 * np.exp(-p2/T) 

    return front_factor * integrand * Lambda(t,p1,p3)

def stepFunction(a,b,c):
    if b*b - 4 * a * c >= 0:
        return 1
    return 0

def J(p1,p2,p3,lower,upper,stepFunction:bool):
    
    xs = np.linspace(lower,upper,100)
    a = 1
    b = 1
    c = 1
    if stepFunction == True:
        integrand = 1/(np.sqrt(-a) * stepFunction(a,b,c))
    else:
        integrand = 1/(np.sqrt(-a))
    
    return 0



