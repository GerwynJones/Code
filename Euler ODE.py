# -*- coding: utf-8 -*-
"""
Created on Tue Oct 04 09:37:15 2016

@author: Admin
"""

from __future__ import division
import numpy as np 
import matplotlib.pyplot as plt


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
           
        
def ODEsolve(Tmax, N, f, method, ic): 
          
    dt = Tmax/N 
    
    t = np.zeros(N)   # defining the time array
    t[0] = ic[0]

    y = np.zeros((2,N))  # defining a y array containging the y values and yp values
    y[0,0] = ic[1]
    y[1,0] = ic[2]
    
    for i in range(0,N-1):
        y[:,i+1]  = method(y[:,i], f, dt , ic[3])
        t[i+1] = t[i] + dt
        
    return y, t 
    
#lambda
w = 2*pi; l = -w*w

#initial times 
ti = 0; Tmax = 1
yi = 1; ypi = 0

#time steps 
N = 1000
n = np.array([N,2*N,4*N])

#initial conditions
ic = np.array([ti, yi, ypi, l])  # initial time, final time, initial y and lambda
 
# solving ODE
R = [ODEsolve(Tmax, N, f, euler, ic) for i,N in enumerate(n)]

def f1(t,l):
    
    y = np.cos(l*t)    
    
    return y # eq of d^2y/dt^2

for i in range(len(R)):
    T = R[i][1]; Y = R[i][0][0]
    plt.subplot(2,1,1)
    plt.plot(T, Y, label='dt')
    plt.xlabel(r'$Time$')
    plt.ylabel(r'$Y$') 
    plt.title('Graph of ODE')
    plt.legend(loc='best') 
    plt.grid() 
    plt.subplot(2,1,2)
    plt.plot(T, f1(T,w)-Y, label='delta t ='  )
    plt.xlabel(r'$Time$')
    plt.ylabel(r'$Error$') 
    plt.legend(loc='best')
    plt.grid() 
    
def ConvergenceTest(Tmax, N, f, ic, method, order):
    
    R = [ODEsolve(Tmax, N, f, euler, ic) for i,N in enumerate(n)]  
    Y1 = R[0][0][0]
    Y2 = R[1][0][0]      
    Y4 = R[2][0][0]
    
    diff1 = (Y1 - Y2[::2])
    diff2 = (2**order)*(Y2[::2] - Y4[::4])
    
    return diff1,diff2

order = 1

a,b = ConvergenceTest(Tmax, n, f, ic, euler, order)

figure()
plt.subplot(2,1,1)
plt.plot(a)
plt.plot(b)
plt.xlabel(r'$Time$')
plt.ylabel(r'$Y$') 
plt.title('Graph of ODE')
plt.legend(loc='best') 
plt.grid() 
plt.subplot(2,1,2)
plt.plot(a/b)
plt.xlabel(r'$Time$')
plt.ylabel(r'$Error$') 
plt.legend(loc='best')
plt.grid() 
    





