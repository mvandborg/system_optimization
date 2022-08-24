# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 15:55:39 2022

@author: madshv
"""

from IPython import get_ipython
get_ipython().magic('reset -sf')
import os
import sys
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
sys.path.insert(0, 'C:/Users/madshv/OneDrive - Danmarks Tekniske Universitet/code')

import numpy as np
import matplotlib.pyplot as plt
from help_functions import dbm
from scipy.optimize import minimize,Bounds
from src.simulation_system import System_simulation_class

# Import physical parameters
from physical_parameters import *

# %% Fiber section parameters
Pp0 = 1.35              # Pump power (W)
Ppr0 = 0.062           # Probe power (W)

L0 = 150e3
L_co = [1]
L_edf = [9]
L_fib =  [150e3]
C = [1]

L_fib[-1] = 300e3-(L0+np.sum(L_co)+np.sum(L_edf)+np.sum(L_fib[0:-1]))
L_tot = L0+np.sum(L_co)+np.sum(L_edf)+np.sum(L_fib)

Fiber_fib0 = Fiber_Sum150
Fiber_pd0 = Fiber_Sum150

Fiber_co = [Fiber_SumULL]
Fiber_edf = [Fiber_edf]
Fiber_fib = [Fiber_SumULL]
Fiber_pd = [Fiber_SumULL]

Nlamnoise = 7
lamnoise_min = 1530*1e-9
lamnoise_max = 1580*1e-9
Nz = 501

# %% Optimization algorithm

def cost_func(X):
    L0 = X[0]
    L_edf[0] = X[1]
    L_fib[-1] = 300e3-(L0+np.sum(L_co)+np.sum(L_edf)+np.sum(L_fib[0:-1]))
    L_tot = L0+np.sum(L_co)+np.sum(L_edf)+np.sum(L_fib)
    
    Sim = System_simulation_class(lam_p,lam_pr,Ppr0,Pp0,L0,Fiber_fib0,Fiber_pd0,Nz,\
                                  Tpulse,T,f_b,FWHM_b,ng,g_b)
    Sim.add_section(L_co,L_edf[0],L_fib[0],Fiber_co[0],\
                    Fiber_edf[0],Fiber_fib[0],Fiber_pd[0],C[0])
    Sim.add_noise(lamnoise_min,lamnoise_max,Nlamnoise)
    Res = Sim.run()
    SNR = Res.calc_SNR()
    Ppr_max = np.max(Res.Ppr)
    
    # Define penalty
    if Ppr_max>0.2:
        SNR = SNR/1000
    print(L0,L_edf[0],SNR)
    return 1/SNR


X_init = [120e3,10]
bnds = Bounds([0,0],[290e3,15])
Res_minimize = minimize(cost_func,X_init,bounds=bnds)






