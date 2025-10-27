"""

Algorithm for solving the Boltzmann equation - using scipy.integrate's solve_ivp

"""
import numpy as np
import sys
import scipy.integrate as sc
from numba import njit
from CollisionOperators import collisions_ann, collisions_sca, eps

################################################################################

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
    
    J = y2 * y2 / E(y2,x)
    Lambda = -f1 * f2 + np.exp(-E(y1,x)-E(y2,x))
    return 1/E(y1,x) * J * Lambda

@njit
def p_annihilation(f1,f2,y1,y2,x):

    J = y2 * y2 / E(y2,x) * (y1 * y1 + y2 * y2) / (x * x)
    Lambda = -f1 * f2 + np.exp(-E(y1,x)-E(y2,x))
    return 1/E(y1,x) * J * Lambda


################################################################################


def solver(x_span,y_span,initial_condition, type:str) -> tuple:
    """
    Solves coupled integro ordinary differential equation 
    """
    @njit
    def trapezoid(y, x):
        
        """Numba-compatible trapezoidal rule."""
        s = 0.0
        for i in range(1, len(y)):
            s += 0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1])
        return s
   

    ys = ys = np.linspace(y_span[0], y_span[1], y_span[2])
    f0 = np.array([initial_condition(y, x_span[0]) for y in ys], dtype=np.float64)

    @njit
    def collision(f1, f3, y1, y3, x,string):

        

        if type == "s-wave annihilation":
            return s_annihilation(f1, f3, y1, y3, x)
        elif type == "p-wave annihilation":
            return p_annihilation(f1,f3,y1,y3,x)
        elif type == "scattering":
            return scattering(f1, f3, y1, y3, x)
        else:
            sys.exit("Unknown collision")


        

    @njit
    def RHS(x:float, fs:list,string):
        
        
    
        N = len(fs)
        rhs = np.empty(N)
        for i in range(N):
            f1 = fs[i]
            y1 = ys[i]
            integrand_values = np.empty(N)
            for j in range(N):
                f3 = fs[j]
                y3 = ys[j]
                integrand_values[j] = collision(f1, f3, y1, y3, x,string)
            rhs[i] = trapezoid(integrand_values, ys)
        
        
        # rhs = np.array([sc.trapezoid([collision(f1,f3,y1,y3,x) for f3,y3 in zip(fs,ys)],ys) for f1,y1 in zip(fs,ys)])
        return rhs
            
    def RHS_wrapper(x,fs):
        
        if type == "annihilation+scattering":
            ann = RHS(x,fs,"s-wave annihilation")
            sca = RHS(x,fs,"scattering")
            return np.add(ann,sca)
        return RHS(x,fs,type)
    
    # Solve the Integro-ODE using Scipy's ODE solver from x0,...,xN
    sol = sc.solve_ivp(fun=RHS_wrapper, t_span=x_span, y0=f0, method="BDF", rtol=1e-6, atol=1e-6)
    xs = sol.t
    fs = sol.y
        
    # compute n_dot

        

    return (xs, ys, fs)


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

def energy_conservation(xs:list,ys:list,fs:list,J:int):
    

    n_result = []

    E_result = []

    

    # integrating the collision operator over y
    for k in range(len(xs)):
        f = fs[:,k]
        x = xs[k]

        c_ann = collisions_ann(f,ys,x,J)
        c_sca = collisions_sca(f,ys,x)

        # computing n_dot
        n_dot = sc.trapezoid([ys[i] * ys[i] * c_sca[i] for i in range(len(ys))],ys)
        
        # computing n
        integrand = [ys[i] * ys[i] * f[i] for i in range(len(ys))]
        n = sc.trapezoid(integrand,ys)
        n_result.append(n_dot/n)

        # computing E_dot
        integrand = [(c_ann[i] + c_sca[i]) * ys[i]*ys[i] * eps(x,ys[i]) for i in range(len(ys))]
        E_dot = sc.trapezoid(integrand,ys)
        E_result.append(E_dot)


    return (n_result,E_result)