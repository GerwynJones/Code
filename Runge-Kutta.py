# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 20:42:24 2016

@author: Gerwyn Jones
"""
from __future__ import division
import numpy as np 
import matplotlib.pyplot as plt

from ODE import ODEsolve,ConvergenceTest

def f(y, l):
    """ A function to call yprime and y """
    return y[1], l*y[0]
    


