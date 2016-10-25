# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 11:53:53 2016

@author: Admin
"""
from __future__ import division
import numpy as np 
import matplotlib.pyplot as plt

a = 0; b = 1
Xc = np.array([a,b])
Nx = 100
Ti = 0; Tf = 5
T = np.array([Ti,Tf])
v = 2

def solver(Xc,Nx,T,v,func):
    
    Ti, Tf = T
    a, b = Xc
    dx = (b-a)/Nx
    dt = dx/v
    Nt = int((Tf-Ti)/dt)
    x= np.linspace(a-dx,b,Nx+2)
    t = np.linspace(Ti,Tf,Nt+1) 
    U = np.zeros((Nx+2, Nt+1))
    Xinit = func(x[1:Nx+1],0.1)
    U[1:Nx+1,0] = Xinit
    U[0,0] = U[Nx,0]
    U[Nx+1,0] = U[1,0]
    	
    for i in range(1,Nt+1):	 	

        U[1:Nx+1,i] = U[0:Nx,i-1] 
        U[0,i] = U[Nx,i]  
        U[Nx+1,i] = U[1,i]

    return x,t,U
    
def func(x,sigma):
    return np.exp(-((x**2)-.5)**2/(2*sigma**2))    

x,t,U = solver(Xc,Nx,T,v,func)

plot(x,U[:,30])