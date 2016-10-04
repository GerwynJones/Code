# -*- coding: utf-8 -*-
"""
Created on Tue Oct 04 09:37:15 2016

@author: Admin
"""
from __future__ import division
import numpy as np 

def f(x):
    return x**2
    


def euler(yn, fy, dt,N):
    ti = yn[0]
    tf = yn[2]
    yi = yn[1]
    
    # define y and t
    
    y = np.zeros(N)    
    t = np.zeros(N)
    y[0] = yi
    t[0] = ti
    
    for i in range(N-1):
        y[i+1] = y[i] + dt*fy(y[i])
        t[i+1] = t[i] + dt
    return y,t
        
#initial conditions
ic = np.array([0,1,1])  # initial time, initial y, final time        
        
N=100 #time steps    
    
dt = (ic[2]-ic[0])/N        
        
y,t = euler(ic,f,dt,N)

plot(t,y)