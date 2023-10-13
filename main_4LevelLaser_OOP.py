"""
Computes steady-state solutions for the rate equations of a four-level laser in
dimensionless units.

References:
    [AT1091] A computer model of laser action in the teaching of computational physics,
             DGH Andrews, DR Tilley,
             Am. J. Phys. 59, 536 (1991),
             https://doi.org/10.1119/1.16815.

Author: OM & KTG
Date: 2023-10-13(published)
"""


import sys
import numpy as np
import matplotlib.pyplot as plt


def plot_time_evolution(t, n1, n2, w, r, method="RK4"):
    r"""plot time evolution of occupation numbers and energy

    Args:
        n1 (array): dimensionless occupation number n1
        n2 (array): dimensionless occupation number n2
        w (array) : dimensionless energy
        r (float) : pump rate
        
    Returns (none):
        Nothing, but generates figure 'fig_Laser_timeEvolution_r%lf.png'%(r)
    """

    params = {
        'figure.figsize': (3.4,0.6*3.4),
        'axes.linewidth': 0.75,
        'lines.linewidth': 0.75,
        'legend.fontsize': 6,
        'axes.labelsize': 7,
        'font.size':  7,
        'xtick.labelsize' :7,
        'ytick.labelsize': 7,
        'font.sans-serif': "Arial"
        }
    
    plt.rcParams.update(params)

    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.11,bottom=0.14,top=0.9, right=0.90)

    ax.plot(t, n1, color="C0", dashes = [3,2], label=r"$n1$")
    ax.plot(t, n2, color="C0", dashes = [1,1], label=r"$n2$")

    ax.tick_params(axis='y', direction='out', length=2, pad=1, right=False)
    ax.set_ylabel("Number density $n$",labelpad=2)
    ax.tick_params(axis='x', direction='out', length=2, pad=1, top=False)
    ax.set_xlim(0,np.max(t))
    ax.set_xlabel("Time $t$",labelpad=3)
    ax.legend(frameon=False,loc=3)

    ax2 = ax.twinx()
    ax2.plot(t, w, color="red", label=r"$w$")
    ax2.tick_params(axis='y', direction='out', length=2, pad=1, left=False)
    ax2.set_ylabel("Energy $w$",labelpad=2)
    ax2.legend(frameon=False,loc=4)

    plt.savefig('fig_Laser_timeEvolution_%s_r%lf.png'%(method,r),dpi=600)


def rate_equations(t, y, pars):
    """Rate equations four-level laser

    Implements rate equations for four-level laser following Ref. [AT1091]

    Args:
        t (float): dummy variable
        y (array, length 3): system variables
        pars (array, length 4): system parameters

    Returns:
        y (array, length 3): rate of change of system variables
    """
    
    # -- UNPACK VARIABLES AND USE MORE TELLING VARIABLE NAMES
    n1, n2, w = y
    
    # -- UNPACK PARAMETERS AND USE MORE TELLING PARAMETER NAMES
    t0, t1, a, r = pars

    # -- RATE EQUATIONS IN DIMENSIONLESS FORM
    dn1dt = -n1/t1 + n2 + w*(n2-n1)
    dn2dt = r - n2 - w*(n2-n1)
    dwdt  = (a*n2 + w*(n2-n1))/(t0*(1-t1)) - w/t0

    return np.asarray([dn1dt, dn2dt, dwdt])


class ODESolver(object):
    """Class called ODESolver for solving ordinary differential equations
    
    Attributes:
        f(function) : ODE function
        res(list)   : An empty list to store the solution
    """
    
    def __init__(self, f):
        # -- INITIALIZE THE ATTRIBUTES OF THE SOLVER-CLASS
        self.f = f
        self.res = []

    def solve(self, t, y):
        """Solves the ODE using provided numerical methods
        
        Args:
            t(array) : Array of time points
            y(array) : Initial values of system variables

        Returns:
            None
        """
        
        f = self.f
        dt = t[1]-t[0]

        self.res.append(y)
        for t_curr in t[:-1]:
            y = self.advance(t_curr, dt, y, f)
            self.res.append(y)

    def advance(self, t_curr, dt, y, f):
        """Advances the solution to next time step

        Args:
            t_curr(float) : current time point
            dt(float)     : time increment
            y(array)      : current values of system variables
            f(function)   : ODE function

        Raises: 
        NotImplementedError : If the method is not implemented in the subclass.
        """
        
        raise NotImplementedError


class RK4(ODESolver):
    """Fourth order Runge-Kutta method

    Implements forward time-stepping via fourth order Runge-Kutta method

    Args:
        t (numpy-array, 1-dim): samples of independent variable with constant mesh width
        dt (float)            : time increment
        y (array, length 2)   : dependent variables
        f (object)            : derivative function of updating of dependent variables

    Returns:
        y (array, length 2)   : updated dependent variables

    Note:
        - derivative function takes 2 parameters in the form f(t,y), where:
            t (numpy-array, 1-dim): samples of independent variable
            y (array, length 2)   : dependent variables
            
        - makes use of efficient array operations enabled by numpy
     """
     
    def advance(self, t, dt, y, f):
        
        y = np.asarray(y)
       
        # -- PRE-COMPUTE STAGES 1 THROUGH 4
        k1 = f(t       , y)
        k2 = f(t + dt/2, y + dt*k1/2 )
        k3 = f(t + dt/2, y + dt*k2/2 )
        k4 = f(t + dt  , y + dt*k3 )
       
        # -- RETURN EXTRAPOLATED RESULT
        return y + dt*(k1 + 2*k2 + 2*k3 + k4)/6


class Euler(ODESolver):
    """ First order Euler Method
        
    Implements forward time-stepping via Euler method
    Args:
        t (numpy-array, 1-dim): samples of independent variable with constant mesh width
        dt (float)            : time increment
        y (array, length 2)   : dependent variables
        f (object)            : derivative function of updating of dependent variables
    
    Returns:
        y (array, length 2)   : updated dependent variables
    
    Note:
        - derivative function takes 2 parameters in the form f(t,y), where:
            t (numpy-array, 1-dim) : samples of independent variable
            y (array, length 2)    : dependent variables
            
        - makes use of efficient array operations enabled by numpy
    """
    
    def advance(self,t,dt,y,f):
          return y + (dt * f(t,y))
    
    
class RK6(ODESolver):
    """Sixth order Runge-Kutta method

    Implements forward time-stepping via sixth order Runge-Kutta method

    Args:
        t (numpy-array, 1-dim): samples of independent variable with constant mesh width
        dt (float)            : time increment
        y (array, length 2)   : dependent variables
        f (object)            : derivative function of updating of dependent variables

    Returns:
        y (array, length 2)   : updated dependent variables

    Note:
        - derivative function takes 2 parameters in the form f(t,y), where:
            t (numpy-array, 1-dim): samples of independent variable
            y (array, length 2)   : dependent variables
            
        - makes use of efficient array operations enabled by numpy
    """ 

    def advance(self,t,dt,y,f):
        a = np.sqrt(21)
        b = 7 - a
        k1 = f(t,                 y)
        k2 = f(t+dt,              y+(dt*k1))
        k3 = f(t+ dt/2,           y+(dt * ((3*k1) + k2)/8))
        k4 = f(t+(2/3 * dt),      y+(dt * (8*k1)+(2*k2)+(8*k3))/27)
        k5 = f(t+(dt/14 * b),     y+(dt * (((9 * a)-21)*k1 - (8*k2*b) + (48*k3*b)-(63-(3*a))*k4)/392))
        k6 = f(t+(dt/14 * (7+a)), y+(dt * ((-1155-(255*a))*k1 - (280+(40*a))*k2 - ((320*a))*k3 + (63+(363*a))*k4 + (2352 + (392*a))*k5)/1960))
        k7 = f(t+dt,              y+(dt * ((330+(105*a))*k1 + (120*k2) + ((280*a)-200)*k3 - ((189*a)-126)*k4 - (686+(126*a))*k5 + (70*b*k6))/180))
    
        return y + (dt*((9*k1)+(64*k3)+(49*k5)+(49*k6)+(9*k7))/180)


def main():

    # -- (1) INITIALIZATION AND DECLARATION OF PARAMETERS 
    t_max = 80
    N_t = 1000
    t0 = 10
    t1 = 0.5
    a = 1.5e-8
    r = 0.5
    t, dt = np.linspace(0, t_max, N_t, endpoint=True, retstep = True)

    # -- (2) PERFORM COMPUTATION 
    # ... set initial condition
    y = np.asarray([0,0,0])   # (n1, n2, w)
    
    # ... get rate equations with proper parameters
    fun = lambda t, y: rate_equations(t, y, (t0, t1, a, r))
    
    # ... instantiate class
    my_solver = RK4(fun)
    my_solver.solve(t, y)

    # -- (3) POSTPROCESS RESULTS
    n1, n2, w = zip(*my_solver.res)
    plot_time_evolution(t, n1, n2, w, r)


if __name__=="__main__":
    main()   