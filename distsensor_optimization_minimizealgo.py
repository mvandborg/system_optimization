# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 15:55:39 2022

@author: madshv
"""
# %%

from IPython import get_ipython
get_ipython().magic('reset -sf')
import os
import sys
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
sys.path.insert(0, 'C:/Users/madshv/OneDrive - Danmarks Tekniske Universitet/code')

import numpy as np
import matplotlib.pyplot as plt
from help_functions import dbm,inv_dbm
from scipy.optimize import minimize,Bounds
from src.simulation_system import System_simulation_class

# Import physical parameters
from physical_parameters import *

# %% Fiber section parameters
Pp0 = inv_dbm(34)        # Pump power (W)
Ppr0 = 0.2              # Probe power (W)

Nsec = 2
L0 = 80e3
L_co = [1,1]
L_edf = [8,8]
L_fib =  [50e3,100e3]
C = [0.1,1]

L_fib[-1] = 300e3-(L0+np.sum(L_co)+np.sum(L_edf)+np.sum(L_fib[0:-1]))
L_tot = L0+np.sum(L_co)+np.sum(L_edf)+np.sum(L_fib)

Fiber_fib0 = Fiber_Sum150
Fiber_pd0 = Fiber_Sum150

Fiber_co = [Fiber_SumULL,Fiber_SumULL]
Fiber_edf = [Fiber_edf,Fiber_edf]
Fiber_fib = [Fiber_SumULL,Fiber_SumULL]
Fiber_pd = [Fiber_SumULL,Fiber_SumULL]

Nlamnoise = 7
lamnoise_min = 1530*1e-9
lamnoise_max = 1580*1e-9
Nz = 501

# %% Optimization algorithm
Norm_fiber = 1e5
Norm_edf = 10
def cost_func(X):
    L0 = X[0]*Norm_fiber
    L_fib[0] = X[1]*Norm_fiber
    L_edf[0] = X[2]*Norm_edf 
    L_edf[1] = X[3]*Norm_edf  
    #C[0] = X[4]
    
    L_fib[Nsec-1] = 300e3-(L0+np.sum(L_co[0:Nsec])+np.sum(L_edf[0:Nsec])+np.sum(L_fib[0:Nsec-1]))
    L_tot = L0+np.sum(L_co[0:Nsec])+np.sum(L_edf[0:Nsec])+np.sum(L_fib[0:Nsec])
    print('L0 =',L0,'Ledf =',L_edf[0:Nsec],'Lfib =',L_fib[0:Nsec],'C =',C[0:Nsec])
    
    Sim = System_simulation_class(lam_p,lam_pr,Ppr0,Pp0,L0,Fiber_fib0,Fiber_pd0,Nz,\
                                  Tpulse,T,f_b,FWHM_b,ng,g_b)
    
    for i in range(0,Nsec):
        Sim.add_section(L_co[i],L_edf[i],L_fib[i],Fiber_co[i],\
                        Fiber_edf[i],Fiber_fib[i],Fiber_pd[i],C[i])
    Sim.add_noise(lamnoise_min,lamnoise_max,Nlamnoise)
    Res = Sim.run()
    SNR = Res.calc_SNR()
    Pb = Res.calc_brillouin_power()
    Ppr_max = np.max(Res.Ppr)
    print('SNR = ',SNR[-1])
    
    err = SNR[-1]
    # Define penalty
    if Ppr_max>0.23:
        err = err*1000
    print('Error = ',err)
    return err
#100e3/Norm_fiber,100e3/Norm_fiber
X_init = [60e3/Norm_fiber,60e3/Norm_fiber,2/Norm_edf,5/Norm_edf,0.1]
bnds = Bounds([0.1,0.1,0.1,0.1,0],[200e3/Norm_fiber,200e3/Norm_fiber,30/Norm_edf,30/Norm_edf,1])
Res_minimize = minimize(cost_func,X_init,bounds=bnds,method='Nelder-Mead')

# %%
Sim = System_simulation_class(lam_p,lam_pr,Ppr0,Pp0,L0,Fiber_fib0,Fiber_pd0,Nz,\
                                  Tpulse,T,f_b,FWHM_b,ng,g_b)
    
for i in range(0,Nsec):
    Sim.add_section(L_co[i],L_edf[i],L_fib[i],Fiber_co[i],\
                    Fiber_edf[i],Fiber_fib[i],Fiber_pd[i],C[i])
Sim.add_noise(lamnoise_min,lamnoise_max,Nlamnoise)
Res = Sim.run()

# %% Parameter sweep
'''
L0_vec = np.linspace(30,150,5)*1e3
Ledf_vec = np.linspace(2,18,5)
C_vec = np.linspace(0.1,1,5)
SNR_mat = np.zeros([len(L0_vec),len(Ledf_vec),len(Ledf_vec),len(L0_vec),len(C_vec)])

Nit = np.product(SNR_mat.shape)
count = 0
for i in range(len(L0_vec)):
    for j in range(len(Ledf_vec)):
        for k in range(len(Ledf_vec)):
            for l in range(len(L0_vec)):
                for m in range(len(C_vec)):
                    X_init = [L0_vec[i]/Norm_fiber,Ledf_vec[j]/Norm_edf,\
                              Ledf_vec[k]/Norm_edf,L0_vec[l]/Norm_fiber,\
                              C_vec[m]]
                    SNR_mat[i,j,k,l,m] = -cost_func(X_init)
                    count += 1
                    print(count/Nit*100)
SNR_mat[SNR_mat<0] = 0
'''
# %% Plotting

idx_max = np.unravel_index(SNR_mat.argmax(),SNR_mat.shape)
idx_max_mat = np.unravel_index(np.argsort(SNR_mat.ravel()),SNR_mat.shape)

SNR_vec = SNR_mat.flatten()
SNR_vec_sorted = np.sort(SNR_vec)

plt.close('all')
fig0,ax0=plt.subplots(constrained_layout=True)
ax0.plot(SNR_vec_sorted)

for i in range(0,10):
    print(L0_vec[idx_max_mat[0][-1-i]],Ledf_vec[idx_max_mat[1][-1-i]],\
          Ledf_vec[idx_max_mat[2][-1-i]],L0_vec[idx_max_mat[3][-1-i]],\
          C_vec[idx_max_mat[4][-1-i]])


# %% Plotting 2

fig1,ax1 = plt.subplots(2,1,constrained_layout=True)
ax1[0].plot(Res.z*1e-3,Res.calc_SNR()*1e-6)
ax1[0].set_ylabel(r'$\Delta f_0$ (MHz)')
ax1[0].set_xlabel(r'$z$ (km)')
ax1[1].plot(Res.z*1e-3,10*np.log10(Res.calc_brillouin_power()*1e3))
ax1[1].set_ylabel(r'Brillouin power (dBm)')
ax1[1].set_xlabel(r'$z$ (km)')



# %%
