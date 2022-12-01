# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def KdeV_spatial(U, h, i, time_step):
    """
    Performs the spatial derivatives of the KdeV PDE
    
    :param U: All the U values at a particular point in time
    :param h: spatial step
    :param i: particular point in space
    :return: Derivative evaluated at that spatial point i
    """
    #for periodic boundary conditions: #questionable

    Uplus1 = U[(i+1) % len(U)]
    
    Umin1  = U[(i-1) % len(U)]
    
    Uplus2 = U[(i+2) % len(U)]
    
    Umin2  = U[(i-2) % len(U)]


    print("+ ",Uplus1)
    print("- ",Umin1)
    #print((Uplus1**2 - Umin1**2))
    #print((Uplus2 - 2*Uplus1 + 2*Umin1 - Umin2))
    

    out = 1/(4*h) * (Uplus1**2 - Umin1**2) * time_step + 1/(2*h**3) * (Uplus2 - 2*Uplus1 + 2*Umin1 - Umin2) * time_step

    
    return out

def analytic(X, a, t):
    """
    Generates the initial condition

    Parameters
    ----------
    X : Array
        The X domain
    a : float
        Soliton parameter
    A : float
        Amplitude
    t : float
        Start time

    Returns
    -------
    Array
    
    The U values of the initial condition function

    """
    return np.array([(12*a**2 * (1/np.cosh(a*(x-4*a**2*t)))**2) for x in X])


def small_sin_input(X):
    X = np.array(X)
        
    out = np.zeros(len(X))
    period = (np.max(X)-np.min(X))/2

    b = 2*np.pi / period
    
    lower_bound_index = round(len(X)/4)
    upper_bound_index = np.argwhere(X <= X[lower_bound_index] + period/2)[-1][0]
    
    for i in range(lower_bound_index, upper_bound_index):
        
        out[i] += np.sin(b*(X[i] + 5))*0.25
    
    return out



def half_sin_input(X):
    X = np.array(X)
    
    
    out = np.zeros(len(X))
    period = np.max(X)-np.min(X)

    b = 2*np.pi / period
    
    lower_bound_index = round(len(X)/4)
    upper_bound_index = np.argwhere(X <= X[lower_bound_index] + period/2)[-1][0]
    
    for i in range(lower_bound_index, upper_bound_index):
        
        out[i] += np.sin(b*(X[i] + 5))
    
    return out

def full_sin_input(X):
    X = np.array(X)
    
    out = np.zeros(len(X))
    period = (np.max(X)-np.min(X))*2

    b = 2*np.pi / period
    
    out = [np.sin(b*(x + 10)) for x in X]
    
    
    
    return np.array(out)



def analytic_solution(X, a=0.5, n_steps=100, dt=0.01):
    
    t = 0
    steps = 0
    
    time_list = [t]
    Un_list = [analytic(X, a, t)]
    
    while steps < n_steps:

        steps += 1
        t = t + dt

        Un = analytic(X, a, t)
        time_list.append(t)
        Un_list.append(Un)
    
    return [time_list, Un_list]
        



def RK2(t_lims, U_0, time_step, rk_alpha, h):
    #initial conditions
    t = t_lims[0]
    steps = 0
    Un = U_0

    #list for storing points in time
    time_list = [t]
    Un_list = [Un]
    Un_snapshot = [Un[2]]
    
    #progress update
    
    print("RK Started")

    while t < t_lims[1]:
        Un_1 = np.zeros(len(U_0))
        #propagating each point through time
        for i in range(0, len(U_0)):
            fa = -1 * KdeV_spatial(Un, h, i)
            U_predictor = Un + (rk_alpha * fa * time_step)

            fb = -1 * KdeV_spatial(U_predictor, h, i)
            #print(1/2*rk_alpha * ((2*rk_alpha -1)*fa + fb) * time_step)
            Un_1[i] = Un[i] + 1/(2*rk_alpha) * ((2*rk_alpha-1)*fa + fb) * time_step
            
        steps += 1
        t = t + time_step
        if steps % 10 == 0:
        
                time_list.append(t)
                Un_list.append(Un_1)
                Un_snapshot.append(Un_1[2])

        Un = Un_1
        
      
    print("RK Complete")

    return [np.array(time_list), np.array(Un_list)]




def RK4(t_lims, U_0, n_steps, time_step, rk_alpha, h):
    #initial conditions
    
    t = t_lims[0]
    #n_steps = round((t_lims[1]-t_lims[0])/time_step)
    
    steps = 0
    Un = U_0

    #list for storing points in time
    time_list = [t]
    Un_list = [Un]
    Un_snapshot = [Un[2]]
    


    print("RK Started")

    #while t < t_lims[1]:
    while steps < n_steps:
        Un_1 = np.zeros(len(U_0))
        #propagating each point through time
        for i in range(0, len(U_0)):
            fa = -1 * KdeV_spatial(Un, h, i, time_step)
            
            U_predictor = Un + (0.5 * fa)
            fb = -1 * KdeV_spatial(U_predictor, h, i, time_step)
            
            U_predictor = Un + (0.5 * fb)
            fc = -1 * KdeV_spatial(U_predictor, h, i, time_step)
                
            U_predictor = Un + fc
            fd = -1 * KdeV_spatial(U_predictor, h, i, time_step)

            Un_1[i] = Un[i] + 1/6 * (fa + 2*fb + 2*fc + fd)
            
        steps += 1
        t = t + time_step
        #if steps % round(n_steps/100) == 0:
        if 1==1:
        
            time_list.append(t)

            Un_list.append(Un_1)
            Un_snapshot.append(Un_1[2])

            
        if steps == round(n_steps/4):
            print("25% Complete")
        elif steps == round(n_steps/2):
            print("50% Complete")
        elif steps == round(n_steps * 3/4):
            print("75% Complete")
            
        Un = Un_1
        
      
    print("RK Complete")

    return [np.array(time_list), np.array(Un_list), Un_snapshot]


#discretised FDE for the normal KdeV equation
def KdeV_spatial_vec(U, h):
    Uplus1 = np.roll(U, -1)
    Umin1  = np.roll(U, 1)
    Uplus2 = np.roll(U, -2)
    Umin2  = np.roll(U, 2)
    
    out = 1/(4*h) * (Uplus1**2 - Umin1**2) + 1/(2*h**3) * (Uplus2 - 2*Uplus1 + 2*Umin1 - Umin2)
    return -1 * out

#discretised FDE for a shockwave equation
def KdeV_spatial_shock(U, h):
    Uplus1 = np.roll(U, -1)
    Umin1  = np.roll(U, 1)
    
    out = 1/(4*h) * (Uplus1**2 - Umin1**2)
    return -1 * out

#discretised FDE for a dampened/diffused shockwave eqn
def KdeV_spatial_damped(U, h, D):
    Uplus1 = np.roll(U, -1)
    Umin1  = np.roll(U, 1)
    
    out = 1/(4*h) * (Uplus1**2 - Umin1**2) - D/(h**2) * (Umin1 - 2*U + Uplus1)
    return -1 * out

#Vectorised RK 4 method    
def RK4_vec(U_0, n_steps=100, dt=0.001, h=0.1):
    #initial conditions
    t = 0
    steps = 0
    Un = U_0
    

    #list for storing points in time
    time_list = [t]
    Un_list = [Un]
    
    print("RK Started")
    while steps < n_steps:
        #propagating each point through time
        fa = KdeV_spatial_vec(Un, h)
        fb = KdeV_spatial_vec(Un + 0.5*fa*dt, h)
        fc = KdeV_spatial_vec(Un + 0.5*fb*dt, h)
        fd = KdeV_spatial_vec(Un + fc*dt, h)
        
        Un = Un + 1/6 * (fa + 2*fb + 2*fc + fd)*dt

        steps += 1
        t = t + dt
        
        Un_list.append(Un)
        time_list.append(t)

      
    print("RK Complete")
    return [np.array(time_list), np.array(Un_list)]

#Vectorised RK 4 Method for Shockwaves
def RK4_shock(U_0, n_steps=100, dt=0.001, h=0.1):
    #initial conditions
    t = 0
    steps = 0
    Un = U_0
    

    #list for storing points in time
    time_list = [t]
    Un_list = [Un]
    
    print("RK Started")
    while steps < n_steps:
        #propagating each point through time
        fa = KdeV_spatial_shock(Un, h)
        fb = KdeV_spatial_shock(Un + 0.5*fa*dt, h)
        fc = KdeV_spatial_shock(Un + 0.5*fb*dt, h)
        fd = KdeV_spatial_shock(Un + fc*dt, h)
        
        Un = Un + 1/6 * (fa + 2*fb + 2*fc + fd)*dt

        steps += 1
        t = t + dt
        
        Un_list.append(Un)
        time_list.append(t)

      
    print("RK Complete")
    return [np.array(time_list), np.array(Un_list)]

#Vectorised RK 4 Method for Shockwaves
def RK4_shock_damped(U_0, n_steps=100, dt=0.001, h=0.1, D=1.0):
    #initial conditions
    t = 0
    steps = 0
    Un = U_0
    

    #list for storing points in time
    time_list = [t]
    Un_list = [Un]
    
    print("RK Started")
    while steps < n_steps:
        #propagating each point through time
        fa = KdeV_spatial_damped(Un, h, D)
        fb = KdeV_spatial_damped(Un + 0.5*fa*dt, h, D)
        fc = KdeV_spatial_damped(Un + 0.5*fb*dt, h, D)
        fd = KdeV_spatial_damped(Un + fc*dt, h, D)
        
        Un = Un + 1/6 * (fa + 2*fb + 2*fc + fd)*dt

        steps += 1
        t = t + dt
        
        Un_list.append(Un)
        time_list.append(t)

      
    print("RK Complete")
    return [np.array(time_list), np.array(Un_list)]


