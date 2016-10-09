# -*- coding: utf-8 -*-
"""
Created on Tue Oct 04 09:37:15 2016

@author: gezer
"""

from __future__ import division
import numpy as np 
import matplotlib.pyplot as plt
from scipy.integrate import odeint


def f(y, l):
    """ A function to call yprime and y """
    return y[1], l*y[0]


def euler(y, f, dt, l):
    """ The Y array has two values so we need to split them up 
    when using euler"""
    f1,f2 = f(y, l)   # f1 gives the value y of the function and f2 gives the value of yprime of the function
    yb,yc = y        # yb gives the initial y values and yc gives initail yp values
  
    # using euler method
    ya = yb + dt*f1  # y value
    yp = yc + dt*f2  # yprime value
    
    return ya,yp
           
        
def ODEsolve(Tmax, dt, f, method, ic): 
          
    N = int (Tmax/dt) 
    
    t = np.zeros(N)   # defining the time array
    t[0] = ic[0]

    y = np.zeros((2,N))  # defining a y array containging the y values and yp values
    y[0,0] = ic[1]
    y[1,0] = ic[2]
    
    for i in xrange(0,N-1):
        y[:,i+1] = method(y[:,i], f, dt , ic[3])
        t[i+1] = t[i] + dt
        
    return y, t 
    
#lambda
w = 2*pi
l = -w*w

#initial times 
ti = 0 
yi = 1
ypi = 0

#time steps 
dt = 2/abs(int(500*l))  

Tmax = 10

#initial conditions
ic = np.array([ti, yi, ypi, l])  # initial time, final time, initial y  
 
# solving ODE
ydt,ta =  ODEsolve(Tmax, dt, f, euler, ic)

ydt2,tb = ODEsolve(Tmax, dt/2, f, euler, ic)

ydt4,tc = ODEsolve(Tmax, dt/4, f, euler, ic)

ya = ydt[0,:]
yb = ydt2[0,:]
yc = ydt4[0,:]

plt.subplot(2,1,1)
plt.plot(ta,ya, label='dt')
plt.plot(tb,yb, label='dt/2')
plt.plot(tc,yc, label='dt/4')
plt.xlabel(r'$Time$')
plt.ylabel(r'$Y$') 
plt.title('Graph of a ODE')
plt.legend(loc='best') 
plt.grid() 

N = int (Tmax/dt) 

def f1(t,l):
    y = cos(l*t)
    #y = np.exp(l*t)    
    
    return y # eq of d^2y/dt^2
        

# splitting the y array into 2 parts
y1 = f1(ta,w)
y2 = f1(tb,w)
y3 = f1(tc,w)

plt.subplot(2,1,2)
plt.plot(ta, ya-y1, label='error dt')
plt.plot(tb, (yb-y2), label='error dt/2')
plt.plot(tc, (yc-y3), label='error dt/4')
plt.xlabel(r'$Time$')
plt.ylabel(r"$Error$") 
plt.legend(loc='best') 
plt.grid() 
plt.show()