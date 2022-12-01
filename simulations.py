# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


#Initial conditions ----------------------------------------------------------
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
        
#FD Spatial derivatives -------------------------------------------------------

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


#butcher tableau coefficients for RKF45
k2_coeffs = [0.25]
k3_coeffs = [3/32, 9/32]
k4_coeffs = [1932/2197, -7200/2197, 7296/2197]
k5_coeffs = [439/216, -8, 3680/513, -845/4104]
k6_coeffs = [-8/27, 2, -3544/2565, 1859/4104, -11/40]

RKF4_coeffs = [25/216, 0, 1408/2565, 2197/4104, -1/5, 0]
RKF5_coeffs = [16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55]



#Embedded RKF (Felhberg) 45 Method
def RKF45_vec(U_0, n_steps=100, dt=0.001, h=0.1, error_tol = 0.01):
    #initial conditions
    t = 0
    steps = 0
    Un = U_0
    

    #list for storing points in time
    time_list = [t]
    Un_list = [Un]
    
    print("RK Started")
    #propagating each point through time
    while steps < n_steps:
        #checking for appropriate step size
        U_diff = 1e5
        while U_diff > error_tol:
        
            k1 = KdeV_spatial_vec(Un, h)*dt
            k2 = KdeV_spatial_vec(Un + k2_coeffs[0]*k1, h)*dt
            k3 = KdeV_spatial_vec(Un + k3_coeffs[0]*k1 + k3_coeffs[1]*k2, h)*dt
            k4 = KdeV_spatial_vec(Un + k4_coeffs[0]*k1 + k4_coeffs[1]*k2 + k4_coeffs[2]*k3, h)*dt
            k5 = KdeV_spatial_vec(Un + k5_coeffs[0]*k1 + k5_coeffs[1]*k2 + k5_coeffs[2]*k3 + k5_coeffs[3]*k4, h)*dt
            k6 = KdeV_spatial_vec(Un + k6_coeffs[0]*k1 + k6_coeffs[1]*k2 + k6_coeffs[2]*k3 + k6_coeffs[3]*k4 + k6_coeffs[4]*k5, h)*dt
            
            U1_Order4 = Un + RKF4_coeffs[0]*k1 + RKF4_coeffs[2]*k3 + RKF4_coeffs[3]*k4 + RKF4_coeffs[4]*k5
            
            U1_Order5 = Un + RKF5_coeffs[0]*k1 + RKF5_coeffs[2]*k3 + RKF5_coeffs[3]*k4 + RKF5_coeffs[4]*k5 + RKF5_coeffs[5]*k6
            
            U_diff = np.max(U1_Order5 - U1_Order4)
            
            s = 0.84 * (error_tol/U_diff)**0.25
            dt = s * dt
            #print(dt)
        
        #passed step size check, completing step
        steps += 1
        t = t + dt
        
        Un = U1_Order5
        Un_list.append(Un)
        time_list.append(t)

      
    print("RK Complete")
    return [np.array(time_list), np.array(Un_list)]




