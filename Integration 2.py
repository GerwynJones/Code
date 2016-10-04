# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 10:32:32 2016

@author: Admin
"""
from __future__ import division
from numpy import polynomial
import numpy as np
from scipy.integrate import quad

from Integration import trap

def f(x):
   return  1/(1+x**2) + 5*( np.exp( -(x - 0.5)**2/(0.01**2) )) 
   
#x**.5 

#range
n=10

i = np.linspace(1,n/2,n)
N = 2**i



a=-5.; b=5.

I = np.zeros(n)
I2 = np.zeros(n)
I4 = np.zeros(n)

for j in range(n):

    I[j] = trap(f,a,b,N[j])
    I2[j] = trap(f,a,b,2*N[j])
    I4[j] = trap(f,a,b,4*N[j])
    
cvg = (I - I2)/(I2 - I4)

#semilogy(N,cvg)

print(I[-1])
#print(pi/4)


Ia = quad(f, a, b)
print(Ia)



#Q3


def GL(F,a,b,n):
    
    fn = polynomial.legendre.leggauss(n) #first array - xi; second array - wi

    I = np.zeros(n)
    
    x = fn[0]    #x
    w = fn[1]    #w

    for j in range(n):
        I[j] = w[j]*F(((b-a)/2)*x[j] + ((a+b)/2))
    
    S1 = ((b-a)/2)*np.sum(I)
    return S1


T = GL(f,a,b,n)


print(T)