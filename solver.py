"""

Algorithm for solving the Boltzmann equation - using scipy.integrate's solve_ivp

"""
import numpy as np
import scipy.integrate as sc
import scipy.linalg as lin

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



def collisions_ann(f,ys,x):

    c = lambda y,x: 1 / (2*np.pi)**3 * 1/(2 * eps(x,y))

    
    integrand = lambda y1,x,f1: [c(y1,x)* yi * yi /(2 * eps(x,yi)) * I(x,y1,yi,f1,f) * F1(x,y1,yi,f1,f[i]) for i,yi in enumerate(ys)]


    # return np.array([sc.simpson(integrand(y,x,f[i]),ys) for i,y in enumerate(ys)])
    return sc.simpson(integrand(ys,x,f),ys)

def collisions_sca(f,ys,x):
    
    c = lambda y,x: 1 / (2*np.pi)**3 * 1/(2 * eps(x,y))

    
    integrand = lambda y1,x,f1: [c(y1,x)* yi * yi /(2 * eps(x,yi)) * H(x,y1,yi,f1,f) * F2(x,y1,yi,f1,f) for i,yi in enumerate(ys)]

    # return np.array([sc.simpson(integrand(y,x,f[i])) for i,y in enumerate(ys)])    
    return sc.simpson(integrand(ys,x,f),ys)





def Hubble(x):
    return 1





def BE_solver(x_span: tuple, y_span: tuple, y_steps, interaction_strength, collisions="both"):


    xmin = x_span[0]

    ymin = y_span[0]
    ymax = y_span[1]

    # define the momentum axis
    ys = np.linspace(ymin,ymax,y_steps)

    # define initial condition
    def initial_condition(y,x):
        return np.exp(-np.sqrt(x*x+y*y))

    f0 = [initial_condition(y,xmin) for y in ys]

    # define Right-hand side
    def rhs(x, f):
        """
        Right-hand side of the BE.
        """
        f = f.reshape(-1) 

        c_ann = collisions_ann(f,ys,x)
        c_sca = collisions_sca(f,ys,x)
        if collisions == "both":                
            interaction_term = c_ann + c_sca
        elif collisions == "scattering":
            interaction_term = c_sca
        elif collisions == "annihilation":
            interaction_term = c_ann
        else:
            interaction_term = 0
        
        return interaction_term



    # solve BE
    sol = sc.solve_ivp(fun=rhs, t_span=x_span, y0=f0, method="BDF", rtol=1e-6, atol=1e-9)


    def energy_conservation(xs,fs):
        
        n_dot = []
        E_dot = []
        print("len(fs[5])",len(fs[5]))
        print("len(xs)",len(xs))
        #c_ann = [collisions_ann(fs[i],ys,xs[i]) for i in range(len(xs))]
        #c_sca = collisions_sca(fs[0],ys,xs[0])

        #n_dot.append(c_sca)
        #   E_dot.append(c_ann)

        return (n_dot,E_dot)
        
    
    xs = sol.t
    fs = sol.y
    conservation = energy_conservation(xs,fs)
    n_dot = conservation[0]
    E_dot = conservation[1]
    return (xs,fs,ys,n_dot,E_dot)
