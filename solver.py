"""

Algorithm for solving the Boltzmann equation - using scipy.integrate's solve_ivp

"""
import numpy as np
import sys
import scipy.integrate as sc
from numba import njit
from CollisionOperators import collisions_ann, collisions_sca, eps, I1, F1

################################################################################


@njit
def test_function(f1,f3,y1,y3,x):
    return - f1 + np.sin(np.pi * y1) * f3

@njit
def E(y,x):
    return np.sqrt(y*y+x*x)

def equilibrium(y,x):
    return np.exp(-E(y,x))

@njit
def scattering(f1,f3,y1,y3,x):

        J = y3 / E(y3,x) * (y1 + y3 - abs(y1 - y3))/y1
        Lambda = -f1 + f3 * np.exp(-(E(y1,x) - E(y3,x)))

        return 1/E(y1,x) * J * Lambda



@njit
def s_annihilation(f1,f2,y1,y2,x):

    # front_factor = 1/E(x,y1)

    # return front_factor * y2 * y2 /(E(x,y2)) * I1(x,y1,y2,f1,f2) * F1(x,y1,y2,f1,f2)
    front_factor = 1/E(y1,x)
    J = y2 * y2 / E(y2,x)
    Lambda = -f1 * f2 + np.exp(-E(y1,x)-E(y2,x))
    return front_factor * J * Lambda

@njit
def p_annihilation(f1,f2,y1,y2,x):

    J = y2 * y2 / E(y2,x) * (y1 * y1 + y2 * y2) / (x * x)
    Lambda = -f1 * f2 + np.exp(-E(y1,x)-E(y2,x))
    return 1/E(y1,x) * J * Lambda

@njit
def trapezoid(y, x):
    s = 0.0
    for i in range(1, len(y)):
        s += 0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1])
    return s

@njit
def collision_operator(x:float, fs:list,ys:list,collision):
        
    N = len(fs)
    rhs = np.empty(N)
    for i in range(N):
        f1 = fs[i]
        y1 = ys[i]
        integrand_values = np.empty(N)
        for j in range(N):
            f3 = fs[j]
            y3 = ys[j]
            integrand_values[j] = collision(f1, f3, y1, y3, x)
        rhs[i] = trapezoid(integrand_values, ys)
    return rhs

################################################################################


def solver(x_span,y_span,initial_condition, type:str) -> tuple:
    """
    Solves coupled integro ordinary differential equation 
    """
   

    ys = ys = np.linspace(y_span[0], y_span[1], y_span[2])
    f0 = np.array([initial_condition(y, x_span[0]) for y in ys], dtype=np.float64)

    @njit
    def collision(f1, f3, y1, y3, x):

        if type == "s-wave annihilation":
            return s_annihilation(f1, f3, y1, y3, x)
        elif type == "p-wave annihilation":
            return p_annihilation(f1,f3,y1,y3,x)
        elif type == "scattering":
            return scattering(f1, f3, y1, y3, x)
        elif type == "annihilation+scattering":
            return s_annihilation(f1,f3,y1,y3,x) + scattering(f1,f3,y1,y3,x)
        elif type == "test":
            return test_function(f1,f3,y1,y3,x)
        else:
            sys.exit("Unknown collision")
  
    def RHS_wrapper(x,fs):
        
        return collision_operator(x,fs,ys,collision)
    
    # Solve the Integro-ODE using Scipy's ODE solver from x0,...,xN
    sol = sc.solve_ivp(fun=RHS_wrapper, t_span=x_span, y0=f0, method="BDF", rtol=1e-6, atol=1e-6)
    xs = sol.t
    fs = sol.y      

    return (xs, ys, fs)


# @njit
# def compute_ndot(xs,fs,ys):
#     N = len(xs)
#     n = len(ys)
    
#     n_dot = np.empty(N)
#     for i in range(N):
#         collisions = np.empty(n)
#         for j in range(n):
#             collisions[j] = collision_operator(xs[i],fs,ys,scattering)
#     return n_dot   



def energy_conservation(x_span,y_span,initial_condition):

    solv = solver(x_span,y_span,initial_condition,"scattering")

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
                integrand_values[j] = scattering(f1, f3, y1, y3, xs[k])
            collision_operator = trapezoid(integrand_values, ys)
            n_dot_integrand[i] = collision_operator * ys[i] * ys[i]
            E_dot_integrand[i] = n_dot_integrand[i] * E(y1, xs[k])
        n_dot[k] = trapezoid(n_dot_integrand,ys) 
        E_dot[k] = trapezoid(E_dot_integrand,ys)
    
    return (n_dot,E_dot,xs)
            











def BE_solver(x_span: tuple, y_span: tuple, y_steps:int,J:int, interaction_strength:float,collision_type="both",initial_condition=lambda y,x: np.exp(-np.sqrt(x*x+y*y))):


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

        c_ann = collisions_ann(fs,ys,x,J)
        c_sca = collisions_sca(fs,ys,x)

        if collision_type == "both":                
            interaction_term = interaction_strength * np.add(c_sca,c_ann)
        elif collision_type == "elastic scattering":
            interaction_term = interaction_strength * c_sca
        elif collision_type == "annihilation":
            interaction_term = interaction_strength * c_ann
        else:
            interaction_term = 0
        
        return interaction_term



    # solve BE
    sol = sc.solve_ivp(fun=rhs, t_span=x_span, y0=f0, method="BDF", rtol=1e-6, atol=1e-9)

    print(f"Solved BE for (type={collision_type},# y-steps={y_steps})")

        
    xs = sol.t
    fs = sol.y
    return (xs,fs,ys)

# def energy_conservation(xs:list,ys:list,fs:list,J:int):
    

#     n_result = []

#     E_result = []

    

#     # integrating the collision operator over y
#     for k in range(len(xs)):
#         f = fs[:,k]
#         x = xs[k]

#         c_ann = collisions_ann(f,ys,x,J)
#         c_sca = collisions_sca(f,ys,x)

#         # computing n_dot
#         n_dot = sc.trapezoid([ys[i] * ys[i] * c_sca[i] for i in range(len(ys))],ys)
        
#         # computing n
#         integrand = [ys[i] * ys[i] * f[i] for i in range(len(ys))]
#         n = sc.trapezoid(integrand,ys)
#         n_result.append(n_dot/n)

#         # computing E_dot
#         integrand = [(c_ann[i] + c_sca[i]) * ys[i]*ys[i] * eps(x,ys[i]) for i in range(len(ys))]
#         E_dot = sc.trapezoid(integrand,ys)
#         E_result.append(E_dot)


#     return (n_result,E_result)