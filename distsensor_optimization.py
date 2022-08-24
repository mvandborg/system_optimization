# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 15:45:34 2022

@author: madshv
"""

"""
import os
import sys
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
"""


from IPython import get_ipython
get_ipython().magic('reset -sf')
import sys
sys.path.insert(0, 'C:/Users/madshv/OneDrive - Danmarks Tekniske Universitet/code')

import numpy as np
from scipy.constants import c
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from numpy import log10,exp,pi
from erbium_model.src.fiberdata_erbium import Erbiumfiber_class
from help_functions import dbm
from src.fiberdata_passive import Passivefiber_class
from src.simulation_system import System_simulation_class
c = c*1e-9              # Unit m/ns

# Import physical parameters
from physical_parameters import *

# %% Fiber section parameters
Pp0 = 1.35              # Pump power (W)
Ppr0 = 0.062           # Probe power (W)

L0 = 150e3
L_co = [1]
L_edf = [9]
L_fib =  [150e3]
C = [0.5]

L_fib[-1] = 300e3-(L0+np.sum(L_co)+np.sum(L_edf)+np.sum(L_fib[0:-1]))
L_tot = L0+np.sum(L_co)+np.sum(L_edf)+np.sum(L_fib)

Fiber_fib0 = Fiber_Sum150
Fiber_pd0 = Fiber_Sum150

Fiber_co = [Fiber_SumULL]
Fiber_edf = [Fiber_edf]
Fiber_fib = [Fiber_SumULL]
Fiber_pd = [Fiber_SumULL]

# %% Simulation
Nlamnoise = 16
lamnoise_min = 1500*1e-9
lamnoise_max = 1600*1e-9
Nz = 501

Sim = System_simulation_class(lam_p,lam_pr,Ppr0,Pp0,L0,Fiber_fib0,Fiber_pd0,Nz,\
                              Tpulse,T,f_b,FWHM_b,ng,g_b)
Sim.add_section(L_co[0],L_edf[0],L_fib[0],Fiber_co[0],\
                Fiber_edf[0],Fiber_fib[0],Fiber_pd[0],C[0])
Sim.add_noise(lamnoise_min,lamnoise_max,Nlamnoise)

Res = Sim.run()
SNR = Res.calc_SNR()


# %% Plotting

plt.close('all')

fig0,ax0 = plt.subplots()
ax0.plot(Res.z*1e-3,dbm(Res.Pp),label='Pump pulse power')
ax0.plot(Res.z*1e-3,dbm(Res.Ppmean),label='Pump mean power')
ax0.plot(Res.z*1e-3,dbm(Res.Ppr),label='Probe power')
ax0.plot(Res.z*1e-3,dbm(Res.Pp_pdf),label='Pump delivery fiber power')
ax0.set_ylabel('Power (dBm)')
ax0.set_xlabel('z (km)')
ax0.legend()


