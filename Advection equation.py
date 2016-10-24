# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 11:53:53 2016

@author: Admin
"""

from __future__ import division
import numpy as np 
import matplotlib.pyplot as plt


### defining the 3 right hand side functions

def rhs1(Vi,dx,Nx):
    Wt = np.zeros(len(Vi))
    Wt[1:Nx+1] = (1/2*dx)*(Vi[2:Nx+2] - Vi[0:Nx])
    return Wt
    
def rhs1(Wi,dx,Nx):
    Vt = np.zeros(len(Wi))
    Vt[1:Nx+1] = (1/2*dx)*(Wi[2:Nx+2] - Wi[0:Nx])
    return Vt
    
def rhs1(Vi,dx,Nx):
    Ut = np.zeros(len(Vi))
    Ut[1:Nx+1] = Vi[1:Nx+1]
    return Ut
    
def func(x, sigma):
    return np.exp(-(x**2)/(2*sigma**2))
    
    	
a = 0; b = 1   
Nx = 100; v = 1
Ti = 0; Tf = 1  
#	space step	size,	where	Nx is	number of space intervals	
dx = (b-a)/Nx  
#     time step size  
dt = dx/v  
#	  number of	time intervals  
Nt = int((Tf-Ti)/dt)  
#  set up of a 1D	 array with	values of the Nx+2 space points  
xgrid = np.linspace(a-dx,b,Nx+2)  
# set	up of	a 1D array with values of the Nt+1	 time points  
tgrid = np.linspace(Ti,Tf,Nt+1)  
# set	up of a 2D	array with	solutions values for all (t,x)  
u = zeros(shape=(Nx+2, Nt+1)) 

#	Set	up	a	1D	array	in	which	the	iniCal	condiCon	is	
inic = np.exp(-(xgrid[1:Nx+1]**2-0.5)**2/(2.*0.1**2))
#	Set	this	array	in	your	soluCon	array	for	t=0		
u[1:Nx+1,0] = inic

#	Set the boundary	conditions up for time t=0	
u[0,0] = u[Nx,0]
u[Nx+1,0] = u[1,0] 

#	loop over time	
for	i	in	range(1,Nt+1):	
    #	Apply	simple “time integrator”	
    u[1:Nx+1,i] = u[0:Nx,i-1]
     # Apply boundary conditions	
    u[0,i] = u[Nx,i]
    u[Nx+1,i] = u[1,i] 
    
plot(u)