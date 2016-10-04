# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 09:11:29 2016

@author: Admin
"""

from __future__ import division
import numpy as np

a = 0
b = 1

#interval
N=10

#array of x values from x=0 to x=1. N points
x = np.linspace(a,b,N+1)

def f(x):
   return 3.*x**2. + 2.*x

#  f(x) = 2x + 3x^2. Call this array "fn".
fn = np.poly1d([3.,2.,0])

dx = (b-a)/(N)

fa = fn(a)
fb = fn(b)

#first and last points
fal = (fa + fb)*(dx/2)

#sum of fn(x)*dx
S = np.sum(fn(x)*dx)

#sum of trap
P = S-fal



def trap(f, a, b, N):
    x = np.linspace(a,b,N+1)
    dx = (b-a)/N
    fa = f(a)
    fb = f(b)
    fal = (fa + fb)*(dx/2.)
    S = np.sum(f(x)*dx)
    P = S-fal
    return P
   
