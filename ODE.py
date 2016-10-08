# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 23:03:12 2015

@author: gerwyn
"""

from __future__ import division 
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

figure()

npts = 1e3
tmax =1
t = np.linspace(0.0, tmax, npts) 

def fvdp(y,t,w):
     
 
    return [y[1], -w*w*y[0] ] # eq of d^2y/dt^2
        
        
yinit = [1,0] # initial values for y and yprime 

p = (2*pi,) # m,A,w values 

y = odeint(fvdp, yinit , t, p) # integrating an ODE

# splitting the y array into 2 parts
y1 = y[:,0] 
yprime = y[:,1]

plt.subplot(2,1,1)
plt.plot(t, y1, 'r', label='position') 
plt.xlabel(r'$Time$')
plt.ylabel(r'$Y$') 
plt.title('Graph of a forced van der Pol oscillator')
plt.legend(loc=1) 
plt.grid() 

plt.subplot(2,1,2)
plt.plot(t, yprime, 'b', label='speed')
plt.xlabel(r'$Y$')
plt.ylabel(r"Y'") 
plt.legend(loc=1) 
plt.grid() 
plt.show()