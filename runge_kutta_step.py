# Code for solving ODEs via Runge-Kutta 4th order method

import numpy as np

def runge_kutta_step(dydt, y, t, h, *args):

    k1 = dydt(y, t, args)
    
    k2 = dydt(y + h*(k1/2), t + h/2, args)

    k3 = dydt(y + h*(k2/2), t + h/2, args)

    k4 = dydt(y + h*k3, t + h, args)

    return (y + (1/6) * h * (k1 + 2*k2 + 2*k3 + k4), t + h)




