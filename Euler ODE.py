# -*- coding: utf-8 -*-
"""
Created on Tue Oct 04 09:37:15 2016

@author: Admin
"""
from __future__ import division
import numpy as np 

def f(l,y):
        
    return y[1], l*y[0]

def euler(y, f,dt,l):
    x = y + dt*f(l,y)  # second term is the differential  
    return x
           
        
def ODEsolve(Tmax, dt, f, method, ic): 
          
    N = int (Tmax/dt) 
    
    y = np.zeros(N)
    t = np.zeros(N)
    y[0] = ic[1]
    t[0] = ic[0]

    for i in xrange(0,N-1):
        y[i+1] = method(y[i],f,dt,ic[2])
        t[i+1] = t[i] + dt
        
    return y,t
    
#lambda
l=-(2*pi)**2    

#initial times 
ti = 0 
yi = 1

#time steps 
dt = 2/(10*abs(l))   

Tmax = 5


#initial conditions
ic = np.array([ti,yi,l])  # initial time, final time, initial y  
 
# solving ODE
y,t =  ODEsolve(Tmax, dt,f,euler,ic)

plot(t,y)


print y[-1]


