# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 11:53:53 2016

@author: Admin
"""
from __future__ import division
import numpy as np 
import matplotlib.pyplot as plt
import time

from PyQt4.QtGui import QApplication # essential for Windows
plt.ion() # needed for interactive plotting

a = -1; b = 1
Xc = np.array([a,b])
Nx = 200
Ti = 0; Tf = 1.2
T = np.array([Ti,Tf])
c = 2

def W1(Vi, dx, Nx):
    Wt = np.zeros(len(Vi))
    Wt[1:Nx+1] = (1/2*dx)*(Vi[2:Nx+2] - Vi[0:Nx])
    return Wt
    
def V1(Wi, dx, Nx):
    Vt = np.zeros(len(Wi))
    Vt[1:Nx+1] = (1/2*dx)*(Wi[2:Nx+2] - Wi[0:Nx])
    return Vt
    
def U1(Vi, dx, Nx):
    Ut = np.zeros(len(Vi))
    Ut[1:Nx+1] = Vi[1:Nx+1]
    return Ut
    
def Euler(Ui, Vi, Wi, W1, V1, U1, dx, dt):

    W = W1(Vi, dx, Nx)     
    V = V1(Wi, dx, Nx)
    U = U1(Vi, dx, Nx)
    
    # using euler method
    Ux = Ui[1:Nx+1] + dt*U  
    Vx = Vi[1:Nx+1] + dt*V  
    Wx = Wi[1:Nx+1] + dt*W
    return Ux, Vx, Wx
    
def func(x, sigma):
    return np.exp(-(x**2)/(2*sigma**2))

def solver(Xc, Nx, T, c, method, func):
    
    Ti, Tf = T
    a, b = Xc
    dx = (b-a)/Nx
    dt = dx/c
    Nt = int((Tf-Ti)/dt)
    x = np.linspace(a-dx,b,Nx+2)
    t = np.linspace(Ti,Tf,Nt+1) 
    U = np.zeros((Nx+2, Nt+1))
    V = np.zeros((Nx+2, Nt+1))  
    W = np.zeros((Nx+2, Nt+1))
    
    #U[x,0]
    Xinit = func(x[1:Nx+1],0.1)
    U[1:Nx+1,0] = Xinit
    W[1:Nx+1,0] = -100*x[1:Nx+1]*Xinit
    
    
    #boundary
    U[0,0] = U[Nx,0]
    U[Nx+1,0] = U[1,0]
    	
    for i in range(1,Nt+1):
        
        U[1:Nx+1,i], V[1:Nx+1,i], W[1:Nx+1,i]  = method(U[0:Nx,i-1], V[0:Nx,i-1], W[0:Nx,i-1], W1, V1, U1, dx, dt)
        
        U[0,i] = U[Nx,i]  
        U[Nx+1,i] = U[1,i]

    return x, t, U

x, t, U = solver(Xc, Nx, T, c, Euler, func)

line1, = plt.plot(x, U[:,0], linewidth=1.0, color='r',label='re') 

for i in range(1,len(t)): # this steps through t values
    line1.set_ydata(U[:,i]) # changes the data for line1 
    plt.draw()

plt.ioff() 
plt.show()

figure()
plt.plot(x, U[:,0])