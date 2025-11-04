import numpy as np
import sys
import scipy.integrate as sc
from numba import njit
from collisions import E, test_function, scattering, s_annihilation, p_annihilation, scattering_
from CollisionOperators import collisions_ann, collisions_sca

def solver(x_span,y_span,Gamma,initial_condition, collision, rtol=1e-6,atol=1e-6) -> tuple:
    """
    Solves coupled integro ordinary differential equation 
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
    # s = 0.0
    # for i in range(1, len(y)):
    #     s += 0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1])
    # return s
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
            

def solver_density(x_span,y_span,Gamma,initial_condition, collision, rtol=1e-6,atol=1e-6):
    """
    Computing the rhs of the integrated Boltzmann equation
    """
    ys = ys = np.linspace(y_span[0], y_span[1], y_span[2])
    n0 = initial_condition(x_span[0])
    
    def RHS_wrapper(x,fs):
        return integrated_operator(x,fs,ys,Gamma,collision)

    # Solve the Integro-ODE using Scipy's ODE solver from x0,...,xN
    sol = sc.solve_ivp(fun=RHS_wrapper, t_span=x_span, y0=n0, method="BDF", rtol=rtol,atol=atol)
    xs = sol.t
    ns = sol.y      

    return (xs, ys, ns)

@njit
def integrated_operator(x:float,fs:list,ys:list,Gamma:float,collision):
    """
    Integrating the collision operator
    """
    collisions = collision_operator(x,fs,ys,Gamma,collision)
    integral = trapezoid(collisions,ys)
    return np.array([integral]).reshape(-1)



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