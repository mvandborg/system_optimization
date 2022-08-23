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

import numpy as np
from scipy.integrate import solve_ivp
from scipy.constants import c,h
from scipy.constants import k as kB
from scipy.integrate import simpson
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from numpy import log10,exp,pi
from erbium_model.fiberdata_erbium import Erbiumfiber_class
from erbium_model.simulation_erbium import Erbium_simulation_class
c = c*1e-9              # Unit m/ns

def dbm(P):
    return 10*log10(P)+30

def propeq(z,P,gr,f_p,f_pr,alpha_p,alpha_pr):
    Pp,Ppr = P
    dPp = -alpha_p*Pp-f_p/f_pr*gr*Pp*Ppr
    dPpr = -alpha_pr*Ppr+gr*Pp*Ppr
    return [dPp,dPpr]

def prop_transmissionfiber(L,Nz,Pp0,Ppr0,gr,f_p,f_pr,alpha_p,alpha_pr):
    zspan = [0,L]
    z = np.linspace(zspan[0],zspan[1],Nz)
    res = solve_ivp(propeq,zspan,[Pp0,Ppr0],method='RK45',
                    t_eval=z,rtol=1e-13,atol=1e-11,args=(gr,f_p,f_pr,alpha_p,alpha_pr))
    Pp = res.y[0]
    Ppr = res.y[1]
    Gfw = Ppr/Ppr[0]
    Gbw = Gfw[-1]/Gfw
    T = 300
    nsp = 1/(1-exp(-h*(f_p-f_pr)/(kB*T)))
    Sase_fw = nsp*h*f_pr*gr*Gfw[-1]*simpson(Pp/Gfw,z)
    Sase_bw = nsp*h*f_pr*gr*Gbw[0]*simpson(Pp/Gbw,z)
    return z,Pp,Ppr,Sase_fw,Sase_bw

def prop_EDF(EDFclass,Nz,Pp0,Ppr0,lam_noise_init):
    z_EDF = np.linspace(0,EDFclass.L,Nz)
    no_of_modes = 2     # Number of optical modes in the fiber
    Sim_EDF = Erbium_simulation_class(EDFclass,no_of_modes,z_EDF)
    Sim_EDF.add_fw_signal(lam_p,Pp0)
    Sim_EDF.add_fw_signal(lam_pr,Ppr0)
    Sim_EDF.add_noise(lam_noise_init)
    res,_ = Sim_EDF.run()
    z_EDF = res.x
    Pp = res.y[0]
    Ppr = res.y[1]
    
    Nlamnoise = len(lam_noise_init)-1
    P_noisefw = res.y[2:2+int(Nlamnoise)]
    P_noisebw = res.y[2+int(Nlamnoise):]
    S_noisefw = np.zeros(P_noisefw.shape)
    S_noisebw = np.zeros(P_noisebw.shape)
    for i in range(0,len(z_EDF)):
        S_noisefw[:,i] = P_noisefw[:,i]/Sim_EDF.BW_noise[0:Nlamnoise]
        S_noisebw[:,i] = P_noisebw[:,i]/Sim_EDF.BW_noise[0:Nlamnoise]
    return z_EDF,Pp,Ppr,S_noisefw,S_noisebw

dir_edf = r'C:/Users/madshv/OneDrive - Danmarks Tekniske Universitet/fiber_data/ofs_edf/'
file_edf = r'LP980_11841.s'

NP_TO_DB = 4.34

no_of_modes = 2         # Number of optical modes in the fiber

Pp0 = 1.35              # Pump power (W)
Ppr0 = 200e-3           # Probe power (W)
lam_p = 1490e-9         # Wavelength (m)
lam_pr = 1550e-9
alpha_p_dB = 0.18       # dB/km
alpha_pr_dB = 0.155     # dB/km
gr = 0.3*1e-3           # Raman gain (1/W/m)

f_p = c/lam_p           # Frequencies (GHz)
f_pr = c/lam_pr

f_delta = f_p-f_pr      

alpha_p = alpha_p_dB/NP_TO_DB*1e-3      # Loss (1/m)
alpha_pr = alpha_pr_dB/NP_TO_DB*1e-3

# Brillouin parameters
Tpulse = 50             # Pulse duration (ns)
T = 300                 # Temperature (K)
f_b = 10.8              # Brillouin frequency shift (GHz)
FWHM_b = 38e-3          # FWHM of Brillouin peak (GHz)
Gamma_b = 2*pi*FWHM_b
vg = c/1.45             # Group velocity (m/ns)
g_b = 0.1470            # Brillouin gain factor(1/W/m)


# Numerical parameters

Nlamnoise = 51
lam_noise_init = np.linspace(1500,1600,Nlamnoise+1)*1e-9

lam_noise = lam_noise_init[0:Nlamnoise]
BW_noise = c*(1/lam_noise_init[0:-1]-1/lam_noise_init[1:])
f_noise = c/lam_noise

idx_noise = np.argmin(np.abs(lam_noise-lam_pr))

Nz = 501

# Fiber lengths (m)
L0 = 100e3
L_co = [1]
L_edf = [10]
L_fib =  [150e3]
C = [1]

L_fib[-1] = 300e3-(L0+np.sum(L_co)+np.sum(L_edf)+np.sum(L_fib[0:-1]))

L_tot = L0+np.sum(L_co)+np.sum(L_edf)+np.sum(L_fib)

Nsec = len(L_co)

X_guess = np.array([L0]+L_edf)
#minimize(err_func,)


# 0: Lead in fiber transmission
Ppmean0 = 1e-6

z_fib,Pp_fib,Ppr_fib,Snoisefw_fib,Snoisebw_fib = prop_transmissionfiber(L0,Nz,0*Ppmean0,Ppr0,gr,f_p,f_pr,alpha_p,alpha_pr)
Pp_pdf_fib = Pp0*exp(-alpha_p*z_fib)

z = z_fib
Pp = Pp_fib
Ppr = Ppr_fib
Ppmean = Ppmean0*exp(-alpha_p*z_fib)
Pp_pdf = Pp_pdf_fib
Snoisebw = [Snoisefw_fib]
Snoisefw = [Snoisebw_fib]
Gsmall = [Ppr_fib[-1]/Ppr_fib[0]]

# To be done: Create fiber classes

for i in range(0,Nsec):
    #2: Co-transmission
    
    Pp_start = Pp_pdf_fib[-1]*C[i]+Pp_fib[-1]*(1-C[i])
    Ppr_start = Ppr_fib[-1]
    z_co,Pp_co,Ppr_co,Snoisefw_co,Snoisebw_co = prop_transmissionfiber(L_co[i],Nz,Pp_start,Ppr_start,gr,f_p,f_pr,alpha_p,alpha_pr)
    Ppmean_co = Pp_start*exp(-alpha_p*z_co)
    Pp_pdf_co = Pp_pdf_fib[-1]*(1-C[i])*exp(-alpha_p*z_co)
    Snoisefw.append(Snoisefw_co)
    Snoisebw.append(Snoisebw_co)
    Gsmall.append(Ppr_co[-1]/Ppr_co[0])
    
    #3: EDF fiber section
    EDF = Erbiumfiber_class.from_ofs_files(dir_edf, file_edf, L_edf[i])
    z_edf,Ppmean_edf,Pprmean_edf,Pnoisefw_edf,Pnoisebw_edf = prop_EDF(EDF,Nz,Ppmean_co[-1],1e-6,lam_noise_init)
    Gp_edf = Ppmean_edf/Ppmean_edf[0]
    Gpr_edf = Pprmean_edf/Pprmean_edf[0]
    Pp_edf = Pp_co[-1]*Gp_edf
    Ppr_edf = Ppr_co[-1]*Gpr_edf
    Pp_pdf_edf = Pp_pdf_co[-1]*exp(-alpha_p*z_edf)
    Snoisebw.append(Pnoisebw_edf[idx_noise,0])
    Snoisefw.append(Pnoisefw_edf[idx_noise,-1])
    Gsmall.append(Gpr_edf[-1])
    
    #4: Transmission fiber section 
    z_fib,Pp_fib,Ppr_fib,Snoisefw_fib,Snoisebw_fib = prop_transmissionfiber(L_fib[i],Nz,Pp_edf[-1],Ppr_edf[-1],gr,f_p,f_pr,alpha_p,alpha_pr)
    Ppmean_fib = Ppmean_edf[-1]*exp(-alpha_p*z_fib)
    Pp_pdf_fib = Pp_pdf_edf[-1]*exp(-alpha_p*z_fib)
    Snoisefw.append(Snoisefw_fib)
    Snoisebw.append(Snoisebw_fib)
    Gsmall.append(Ppr_fib[-1]/Ppr_fib[0])
    
    
    z = np.concatenate([z,z[-1]+z_co])
    z = np.concatenate([z,z[-1]+z_edf])
    z = np.concatenate([z,z[-1]+z_fib])
    Pp = np.concatenate([Pp,Pp_co,Pp_edf,Pp_fib])
    Ppr = np.concatenate([Ppr,Ppr_co,Ppr_edf,Ppr_fib])
    Ppmean = np.concatenate([Ppmean,Ppmean_co,Ppmean_edf,Ppmean_fib])
    Pp_pdf = np.concatenate([Pp_pdf,Pp_pdf_co,Pp_pdf_edf,Pp_pdf_fib])


# Post processing
Gacc = 1
Snoisebw_total = Snoisebw[0]*Gacc
for i in range(0,len(Gsmall)-1):
    Gacc = Gacc*Gsmall[i]
    Snoisebw_total = Snoisebw_total+Snoisebw[i+1]*Gacc


Epr = Ppr*Tpulse        # Pulse energy (nJ)
P_b = kB*T*Gamma_b*f_pr/(4*f_b)*g_b*vg*Epr#*exp(1/2*g_b*vg*Epr)

P_b_end = P_b[-1]*Gacc*Gsmall[-1]

SNR = P_b_end/(Snoisebw_total*FWHM_b)



# %% Plotting

plt.close('all')

fig0,ax0 = plt.subplots()
ax0.plot(z*1e-3,dbm(Pp),label='Pump pulse power')
ax0.plot(z*1e-3,dbm(Ppmean),label='Pump mean power')
ax0.plot(z*1e-3,dbm(Ppr),label='Probe power')
ax0.plot(z*1e-3,dbm(Pp_pdf),label='Pump delivery fiber power')
ax0.set_ylabel('Power (dBm)')
ax0.set_xlabel('z (km)')
ax0.legend()

fig1,ax1 = plt.subplots(constrained_layout=True)
ax1.plot(lam_noise*1e9,dbm(Pnoisebw_edf[:,0]))
ax1.set_xlabel('Wavelength (nm)')
ax1.set_ylabel('Noise power (dBm/Hz)')

