import numpy as np

def integrate(ys, xmin, xmax):
    """
    Implementation of Simpson's method to solve 1-dimensional integrals
    """
    # Requirement: Number of interals is odd
    N = len(ys) - 1

    if N % 2 == 0:
        print("Number of steps must be odd and (N-1)/2 must be even")
        sys.exit()
    
    dx = (xmax - xmin) / N

    area = ys[0] + ys[-1] + sum([2 * ys[n] if n % 2 == 0 else 4 * ys[n] for n in range(1,N)])

    return (dx / 3.0) * area


## Various functions that we need to integrate the right-hand side of the Boltzmann equation
def E(p):
    m = 1
    return np.sqrt(p*p+m*m)


def F1(fp,fq,p,q,T):
    return -fp * fq + np.exp(-(E(p)+E(q))/T)

def F2(fp,fq,p,q):
    return -fp + fq * np.exp(-E(p)-E(q))

def I(fp1,fp2,p1,p2,T):
    G = 1
    return 0.5 * G * G * F1(fp1,fp2,p1,p2,T)

def H(fp1,fp3,p1,p3):
    G = 1
    # return G * G * 2 /(p1 * p3) (np.exp(-(p1+p3)/2))
    return G * G * 2 /(p1 * p3) * (np.exp(-abs(p1-p3)/2)-np.exp(-(p1+p3)/2)) * F2(fp1,fp3,p1,p3) 


# Collision integrals
def C_ann(p, qs, fp, fqs, pmin, pmax,t) -> float:
    """
    1D collision operator for annihilation
    """

    T = 1/np.sqrt(t)
    y = []

    for i in range(len(qs)):
        integrand = qs[i] * qs[i] / (2 * E(qs[i])) * I(fp,fqs[i],p,qs[i],T) * F1(fp,fqs[i],p,qs[i],T)
        y.append(integrand)
    
    return 1/ (2 * np.pi)**2 * 1/(2 * E(p)) * integrate(y,pmin,pmax)

def C_scatter(p, qs, fp, fqs, pmin, pmax) -> float:
    """
    1D collision operator for elastic scattering
    """
    
    y = []

    for i in range(len(qs)):
        integrand = qs[i] * qs[i] / (2 * E(qs[i])) * H(fp,fqs[i],p,qs[i]) * F2(fp,fqs[i],p,qs[i])
        y.append(integrand)
    
    return 1/ (2 * np.pi)**2 * 1/(2 * E(p)) * integrate(y,pmin,pmax)


#Hubble parameter and equilibrium distribution function
def Hubble(t):
    """
    Description: The Hubble Paramter
    """
    return 1/(t)

def dist_eq(p,t):
    """
    Desciption: Distribution function for full thermal equilibrium
    """
    T = 1/np.sqrt(t)
    return np.exp(-E(p)/T)


# BE equation solver
def BE_solver(ts:list,ps:list) -> list:
    """
    Description: Numerical solver of the Boltzmann equation.

    parameters: Grid/2-dimensional list of time x momenta intervals
    
    Return: list of distribution function values for time x momenta grid.
    """
    print("Initiating solving the Boltzmann Equation non-dimensionless")
    
    # Defining the grid with shape len(ps) x len(ts)
    grid = np.zeros(shape=(len(ps),len(ts)))

    for i in range(len(ps)):
        grid[i][0] = dist_eq(ps[i],ts[0])

    for i in range(len(ts)-1):

        dt = ts[i+1] - ts[i] # time step
        dp = abs(ps[2] - ps[1]) # All momenta time steps are uniform

        for j in range(len(ps)):

            f = grid[j][i]
            p = ps[j]
            if j < len(ps) - 1:
                df_dp = (grid[j+1][i] - f) / dp
            else:
                df_dp = (f - grid[j-1][i]) / dp  
            H_t = Hubble(ts[i])

            fps = [grid[k][i] for k in range(len(ps))]

            I1 = C_ann(p,ps,f,fps,ps[0],ps[-1],ts[i])
            I2 = C_scatter(p,ps,f,fps,ps[0],ps[-1])

            f_new = f + dt * (p * H_t * df_dp + I1 + I2)

            grid[j][i+1] = f_new
    print("BE solver non-dimensionless: complete")
    return grid
