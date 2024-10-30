# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 15:45:34 2022

@author: madshv
"""

import os
import sys

# Insert current path
this_dir = os.path.dirname(__file__)
sys.path.append(this_dir)

# Insert path to erbium model, loaded from config file
from src.help_functions import load_config
erbium_path = load_config().get('erbium_model_path', 'default_value_if_not_set')
sys.path.append(erbium_path)

import numpy as np
import matplotlib.pyplot as plt
from src.help_functions import dbm
from src.simulation_system import System_simulation_class
from src.fiberdata_passive import Passivefiber_class
from erbium_model.src.fiberdata_erbium import Erbiumfiber_class

# %% Fiber section parameters
Pp0 = 2.2              # Pump power (W)
Ppr0 = 0.2           # Probe power (W)

T0pr = 100e-9
lam_pr = 1.55e-6
lam_p = 1.48e-6

lam_arr = np.array([lam_p,lam_pr])

L0 = 140e3
L_co = [1]
L_edf = [22.2]
L_fib =  [160e3]
C = [1]
Nsec = len(L_co)

L_fib[-1] = 300e3-(L0+np.sum(L_co)+np.sum(L_edf)+np.sum(L_fib[0:-1]))
L_tot = L0+np.sum(L_co)+np.sum(L_edf)+np.sum(L_fib)

# Definitions of passive fiber classes
fiberdata_path = os.path.join(this_dir,'fiber_data')
Fiber_Sum150 = Passivefiber_class.from_data_sheet(
    fiberdata_path,
    'Sumitomo_150.json',
    lam_arr)
Fiber_SumULL = Passivefiber_class.from_data_sheet(
    fiberdata_path,
    'Sumitomo_ULL.json',
    lam_arr)

# Definition of EDF class
dir_edf = erbium_path+r'/erbium_model/fiberdata_ofs/'
file_edf = r'LP980_22841_labversion.s'
Fiber_edf = Erbiumfiber_class.from_ofs_files(dir_edf, file_edf)
#'LP980_22841_labversion','LP980_11841'

Fiber_fib0 = Fiber_Sum150
Fiber_pd0 = Fiber_Sum150

Fiber_co = [Fiber_SumULL,Fiber_SumULL]
Fiber_edf = [Fiber_edf,Fiber_edf]
Fiber_fib = [Fiber_SumULL,Fiber_SumULL]
Fiber_pd = [Fiber_SumULL,Fiber_SumULL]

# %% Simulation
Nlamnoise = 16
lamnoise_min = 1500e-9
lamnoise_max = 1600e-9
Nz = 501

Sim = System_simulation_class(lam_p,lam_pr,Ppr0,Pp0,L0,Fiber_fib0,Fiber_pd0,Nz,\
                              T0pr,T,f_b,FWHM_b,ng,g_b)
for i in range(0,Nsec):
    Sim.add_section(L_co[i],L_edf[i],L_fib[i],Fiber_co[i],\
                    Fiber_edf[i],Fiber_fib[i],Fiber_pd[i],C[i])
Sim.add_noise(lamnoise_min,lamnoise_max,Nlamnoise)

Res = Sim.run()
SNR = Res.calc_SNR()

# %% Plotting

plt.close('all')

fig0,ax0 = plt.subplots()
#ax0.plot(Res.z*1e-3,dbm(Res.Pp),label='Pump pulse power')
ax0.plot(Res.z*1e-3,dbm(Res.Ppmean),label='Pump mean power')
ax0.plot(Res.z*1e-3,dbm(Res.Ppr),label='Probe power')
ax0.plot(Res.z*1e-3,dbm(Res.Pp_pdf),label='Pump delivery fiber power')
ax0.set_ylabel('Power (dBm)')
ax0.set_xlabel('z (km)')
ax0.legend()


