"""

Algorithm for solving the Boltzmann equation - using scipy.integrate's solve_ivp

"""
import numpy as np
import scipy.integrate as sc
from CollisionOperators import collisions_ann, collisions_sca, eps


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
