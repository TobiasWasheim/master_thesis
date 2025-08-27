"""

Algorithm for solving the Boltzmann equation - using scipy.integrate's solve_ivp

"""
import numpy as np
import scipy.integrate as sc
import scipy.linalg as lin

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
    return -f1 + f2 * np.exp(-eps(x,y1)-eps(x,y2))

def I(x:float,y1:float,y2:float,f1:float,f2:float) -> float:
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
    return 2 /(y1 * y2) * (np.exp(- abs(y1 - y2)/2) - np.exp(-(y1 + y2)/2)) * F2(x, y1, y2, f1, f2)



def collisions_ann(f,ys,x):
    """ 
    
    """
    # define functions I and F1 
    I1_Matrix = []
    # f = f.shape(-1)

    c = lambda y,x: 1 / (2*np.pi)**3 * 1/(2 * eps(x,y))

    
    integrand = lambda y1,x,f1: [c(y1,x)* yi * yi /(2 * eps(x,yi)) * I(x,y1,yi,f1,f) * F1(x,y1,yi,f1,f[i]) for i,yi in enumerate(ys)]


    return sc.trapezoid(integrand(ys,x,f),ys)

def collisions_sca(f,ys,x):
    # f = f.shape(-1)
    
    c = lambda y,x: 1 / (2*np.pi)**3 * 1/(2 * eps(x,y))

    
    integrand = lambda y1,x,f1: np.array([c(y1,x)* yi* yi/(2 * eps(x,yi)) * H(x,y1,yi,f1,f) * F2(x,y1,yi,f1,f) for i,yi in enumerate(ys)])

    return sc.trapezoid(integrand(ys,x,f),ys)


def collision_ann(x:float,y:float,ys:list,f:float,fs:list):

    front_factor = 1 / (2*np.pi)**3 * 1/(2 * eps(x,y))
    
    integrand_values = np.array([front_factor* yi* yi /(2 * eps(x,yi)) * I(x,y,yi,f,fi) * F1(x,y,yi,f,fi) for yi,fi in zip(ys,fs)])

    integral = sc.trapezoid(integrand_values, x = ys)

    return integral

def collision_sca(x:float,y:float,ys:list,f:float,fs:list) -> float:
    
    front_factor = 1 / (2*np.pi)**3 * 1/(2 * eps(x,y))

    integrand_values = np.array([front_factor * yi * yi /(2 * eps(x,yi)) * H(x,y,yi,f,fi) * F2(x,y,yi,f,fi) for yi,fi in zip(ys,fs)])

    integral = sc.trapezoid(integrand_values, x = ys)

    return integral





def BE_solver(x_span: tuple, y_span: tuple, y_steps, interaction_strength, collisions="both",initial_condition=lambda y,x: np.exp(-np.sqrt(x*x+y*y))):


    xmin = x_span[0]

    ymin = y_span[0]
    ymax = y_span[1]

    # define the momentum axis
    ys = np.linspace(ymin,ymax,y_steps)

    # define initial distribution
    f0 = [initial_condition(y,xmin) for y in ys]

    # define Right-hand side
    def rhs(x, fs):
        """
        Right-hand side of the BE.
        """

        c_ann = np.array(collisions_ann(fs,ys,x))
        c_sca = np.array(collisions_sca(fs,ys,x))

        # c_ann = np.array([collision_ann(x,y,ys,f,fs)/(x*x) for y,f in zip(ys,fs)])
        # c_sca = np.array([collision_sca(x,y,ys,f,fs)/(x*x) for y,f in zip(ys,fs)])
        if collisions == "both":                
            interaction_term = interaction_strength * (c_ann + c_sca)
        elif collisions == "scattering":
            interaction_term = interaction_strength * c_sca
        elif collisions == "annihilation":
            interaction_term = interaction_strength * c_ann
        else:
            interaction_term = 0
        
        return interaction_term



    # solve BE
    sol = sc.solve_ivp(fun=rhs, t_span=x_span, y0=f0, method="BDF", rtol=1e-6, atol=1e-9)


        
    
    xs = sol.t
    fs = sol.y
    return (xs,fs,ys)

def energy_conservation(x:float,ys:list,f:float,fs:list):
    

    # number density conservation for elastic scattering
    n_dot = sc.trapezoid([y * y * collision_sca(x,y,ys,f,fs) for y in ys],ys)
    n = sc.trapezoid([yi*yi*fi for yi, fi in zip(ys,fs)],ys)

    # energy conservation 
    E_dot = sc.trapezoid([yi * yi * eps(x,yi) * (collision_ann(x,yi,ys,f,fs) + collision_sca(x,yi,ys,f,fs))for yi in ys],ys)


    return (n_dot/n,E_dot)
