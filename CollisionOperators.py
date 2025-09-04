import numpy as np
import scipy.integrate as sc

def eps(x:float,y:float) -> float:
    """
    Dimensionless energy (Energy/Temperature)
    """
    return np.sqrt(x*x + y*y)


def F1(x:float,y1:float, y2:float,f1:float,f2:float) -> float:
    """
    Part of integrand of the annihilation collision operator's integral

    @params: \n
    x:  -> time variable \n
    y1: -> momentum variable \n
    y2: -> momentum integration variable \n
    f1: -> distribution function value at (y1,x) \n
    f2: -> distribution function value at (y2,x)
    """
    return -f1 * f2 + np.exp(-eps(x,y1) - eps(x,y2))


def F2(x:float,y1:float,y2:float,f1:float,f2:float) -> float:
    """
    Part of integrand of the elastic scattering collision operator's integral

    @params: \n
    x:  -> time variable \n
    y1: -> momentum variable \n
    y2: -> momentum integration variable \n
    f1: -> distribution function value at (y1,x) \n
    f2: -> distribution function value at (y2,x)
    """
    return -f1 + f2 * np.exp(eps(x,y1) - eps(x,y2))

def I1(x:float,y1:float,y2:float,f1:float,f2:float) -> float:
    """
    Part of integrand of the annihilation collision operator's integral

    @params: \n
    x:  -> time variable \n
    y1: -> momentum variable \n
    y2: -> momentum integration variable \n
    f1: -> distribution function value at (y1,x) \n
    f2: -> distribution function value at (y2,x)
    """
    return 1 / 2 * F1(x,y1,y2,f1,f2)

def I2(x:float,y1:float,y2:float,f1:float,f2:float) -> float:
    """
    Part of integrand of the annihilation collision operator's integral

    @params: \n
    x:  -> time variable \n
    y1: -> momentum variable \n
    y2: -> momentum integration variable \n
    f1: -> distribution function value at (y1,x) \n
    f2: -> distribution function value at (y2,x)
    """
    return I1(x,y1,y2,f1,f2) * (y1 * y1 + y2 * y2) / (x * x)


def H(x:float,y1:float,y2:float,f1:float,f2:float) -> float:
    """
    Part of integrand of the elastic scattering collision operator's integral

    @params: \n
    x:  -> time variable \n
    y1: -> momentum variable \n
    y2: -> momentum integration variable \n
    f1: -> distribution function value at (y1,x) \n
    f2: -> distribution function value at (y2,x)
    """
    return 2 /(y1 * y2) * (np.exp(-abs(y1 - y2)/2) - np.exp(-(y1 + y2)/2)) * F2(x, y1, y2, f1, f2)



def collisions_ann(fs:list,ys:list,x:float,J:int) -> list:
    """ 
    Collision operator for the annihilation process when J = 0 or J = 1

    @params: \n
    f:  -> list of distribution function values at time x for different y-values \n
    x:  -> time variable \n
    ys: -> list of momentum values
    """

    front_factor = lambda y,x: 1 / (2*np.pi)**3 * 1/(2 * eps(x,y))

    if J == 0:
        integrands = np.array([
            [front_factor(y1,x) * y2 * y2 /(2 * eps(x,y2)) * I1(x,y1,y2,f1,f2) for y2,f2 in zip(ys,fs)] for y1,f1 in zip(ys,fs)
            ])
    else:
        integrands = np.array([
            [front_factor(y1,x) * y2 * y2 /(2 * eps(x,y2)) * I2(x,y1,y2,f1,f2) for y2,f2 in zip(ys,fs)] for y1,f1 in zip(ys,fs)
            ])
    
    integrals = np.array([sc.trapezoid(integrand,ys) for integrand in integrands])
    return integrals
    
    # if J == 0:
    #     integrand = lambda y1,x,f1: [front_factor(y1,x)* yi * yi /(2 * eps(x,yi)) * I1(x,y1,yi,f1,fs) * F1(x,y1,yi,f1,fs[i]) for i,yi in enumerate(ys)]
    # else:
    #     integrand = lambda y1,x,f1: [front_factor(y1,x)* yi * yi /(2 * eps(x,yi)) * I2(x,y1,yi,f1,fs) * F1(x,y1,yi,f1,fs[i]) for i,yi in enumerate(ys)]
    
    # return sc.trapezoid(integrand(ys,x,fs),ys)

def collisions_sca(fs:list,ys:list,x:float) -> list:
    """ 
    Collision operator for the elastic scattering process

    @params: \n
    f:  -> list of distribution function values at time x for different y-values \n
    x:  -> time variable \n
    ys: -> list of momentum values
    """
    front_factor = lambda y,x: 1 / (2*np.pi)**3 * 1/(2 * eps(x,y))

    integrands = np.array([[front_factor(y1,x) * y2 * y2 /(2 * eps(x,y2)) * H(x,y1,y2,f1,f2) for y2,f2 in zip(ys,fs)] for y1,f1 in zip(ys,fs)])
    integrals = np.array([sc.trapezoid(integrand,ys) for integrand in integrands])
    return integrals

    # integrand = lambda y1,x,f1: [front_factor(y1,x) * yi * yi /(2 * eps(x,yi)) * H(x,y1,yi,f1,fs) * F2(x,y1,yi,f1,fs[i]) for i,yi in enumerate(ys)]
    # return sc.trapezoid(integrand(ys,x,fs),ys)