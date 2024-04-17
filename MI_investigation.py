# %% Import modules
import os
import sys
this_dir = os.path.dirname(__file__)
sys.path.append(this_dir)

import numpy as np
from numpy import sqrt,exp
from src.fiberdata_passive import Passivefiber_class
from src.simulation_system import Simulation_pulsed_sections_fiber
from src.help_functions import PSD_dbmGHz2dbmnm, PSD_dbmnm2dbmGHz,dbm, norm_fft
import src.help_functions as hf
import matplotlib.pyplot as plt
from scipy.fft import fftshift

# %% Define propagation fibers
def A0_func(t,T0,Ppeak0):
    return sqrt(Ppeak0)*exp(-(2*t/T0)**22)

# Plotting functions
def plot_power_vs_tf(S,idx_z,dbscale=False):
    A = S.A[:,idx_z]
    AF = norm_fft(A,S.dt)
    ESD = np.abs(AF)**2
    PSD = hf.ESD2PSD(ESD,S.Tmax)  
    PSD_dbmghz = 10*np.log10(PSD*1e3)
    PSD_dbmnm = PSD_dbmGHz2dbmnm(PSD_dbmghz,1550,3e8)
     
    fig,ax = plt.subplots(1,2,constrained_layout=True)
    
    ax[0].plot(S.t,np.abs(A)**2)
    ax[0].set_ylabel('Power (W)')
    
    if dbscale==False:
        ax[1].plot(fftshift(S.f),PSD)
        ax[1].set_ylabel('PSD (nJ/GHz)')
    else:
        ax[1].plot(fftshift(S.f),PSD_dbmnm)
        ax[1].set_ylabel('Power (dBm/nm)')
    ax[1].set_xlabel('Frequency (GHz)')
    ax[1].grid()
    ax[0].set_xlabel('Time (ns)')
    ax[0].grid()

def plot_Anoise(S):
    AF = norm_fft(S.Anoise,S.dt)
    P = np.abs(S.Anoise)**2
    PF = np.abs(AF)**2
        
    fig,ax = plt.subplots(1,2,constrained_layout=True)    
    ax[0].plot(S.t,np.abs(S.Anoise)**2)
    ax[0].set_xlabel('Time (ns)')
    ax[0].set_ylabel('Power (W)')
    ax[0].grid()
    
    ax[1].plot(S.f_sh,PF)
    ax[1].set_xlabel('Frequency (GHz)')
    ax[1].set_ylabel('ESD (nJ/GHz)')
    ax[1].grid()

def plot_power_vs_f(S,idx_z_vec):
    
     
    fig,ax = plt.subplots(constrained_layout=True)
    
    for idx_z in idx_z_vec:
        A = S.A[:,idx_z]
        AF = norm_fft(A,S.dt)
        # Normal ESD
        ESD = np.abs(AF)**2
        # Apply moving average filter
        Nav = 200
        ESD = np.convolve(np.ones(Nav),ESD,mode='same')/Nav
        PSD = hf.ESD2PSD(ESD,S.Tmax)  
        PSD_dbmghz = 10*np.log10(PSD*1e3)
        PSD_dbmnm = PSD_dbmGHz2dbmnm(PSD_dbmghz,1550,3e8)
            
        lab = 'z='+str(int(S.z[idx_z]*1e-3))+' km'
    
        ax.plot(fftshift(S.f),PSD_dbmnm,label=lab)
    ax.set_xlabel('Frequency (GHz)')
    ax.set_ylabel('Power (dBm/nm)')
    ax.grid()
    ax.legend()

# %% Run simulation

# Directory for saving the data
savedir = this_dir+r'\data\MI_test\sec1_SMF28'

L = 100e3               # Fiber length (km)
T0 = 100                # Pulse length (ns)
lam_p = 1455e-9         # Wavelength (m)
lam_pr = 1550e-9
lam_arr = np.array([lam_p,lam_pr])
Ppeak0 = 200e-3
PSD_noise_dbmnm = -30
PSDnoise_dbmGHz = PSD_dbmnm2dbmGHz(PSD_noise_dbmnm,lam_pr*1e9,2.998e8)

fiberdata_path = os.path.join(this_dir, 'fiber_data')
Fiber = Passivefiber_class.from_data_sheet( fiberdata_path,
                                            'OFS_SCUBA150.json',
                                            lam_arr)

Fiber_SMF28 = Passivefiber_class.from_data_sheet( fiberdata_path,
                                            'Corning_SMF28.json',
                                            lam_arr)

Tmax = T0*7             # Simulation window size (ns)
N = 2**16
t = np.linspace(-Tmax/2,Tmax/2,N)
Nz_save = 21
Nsec = 1

# %% Run simulation

A0 = A0_func(t, T0, Ppeak0)
S = Simulation_pulsed_sections_fiber(t, A0, L, Nz_save, Fiber_SMF28, 
                                     PSDnoise_dbmGHz, Nsec)
z, A = S.run()


# %% Plotting 

plt.close('all')
plot_power_vs_tf(S,0,dbscale=True)
plot_Anoise(S)
plot_power_vs_f(S,[0,2,4,6,8,10])

# %%
