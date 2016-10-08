# -*- coding: utf-8 -*-
"""
Created on Tue Oct 04 09:37:15 2016

@author: Admin
"""
from __future__ import division
import numpy as np 
from Integration2 import CVG

def o(l,x):
    # gives the original function    
    
    return np.exp(l*x)

def fp(l,x):
    # gives the differential function    
    
    return l*np.exp(l*x)



def euler(o, f1,f2, t,l,dt):
    x = o(l,t) + dt*f1(l,t)  # second term is the differential
    z = x + dt*f2(l,t)  
    return x,z
           
        
def ODEsolve(N,o, f1,f2, method, ic): 
          
    dt = (ic[1]-ic[0])/N 
    
    y = np.zeros(N)   
    yp = np.zeros(N)
    t = np.zeros(N)
    y[0] = ic[2]
    yp[0] = ic[3]
    t[0] = ic[0]

    for i in arange(0,N-1):
        y[i+1],yp[i+1] = method(o,f1,f2,t[i],l,dt)
        t[i+1] = t[i] + dt
        
    return y,yp,t
    
# lambda    
l=-5

#initial times 
ti = 0 
tf = 1

#time steps 
N=1000   


#initial conditions
ic = np.array([ti,tf,o(l,ti),0])  # initial time, final time, initial y  
 
# solving ODE
y,yp,t =  ODEsolve(N,o,fp,fpp,euler,ic)

plot(t,y)
plot(t,yp)

print y[-1]


"""
y2,t2 = ODEsolve(N/2,fp,euler,ic)
y4,t4 = ODEsolve(N/4,fp,euler,ic)

I = y
I2 = y2
I4 = y4


CVG = (I - I2)/(I2 - I4)

plot(t2,y2) """