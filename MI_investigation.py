# %% Import modules
import os
import sys
this_dir = os.path.dirname(__file__)
sys.path.append(this_dir)

import numpy as np
from numpy import sqrt,exp
from src.fiberdata_passive import Passivefiber_class
from src.simulation_system import Simulation_pulsed_sections_fiber
from src.help_functions import PSD_dbmGHz2dbmnm, PSD_dbmnm2dbmGHz,dbm, norm_fft, norm_ifft
import src.help_functions as hf
import matplotlib.pyplot as plt
from scipy.fft import fftshift
from MI_investigation_sweep import get_ASE_noise_WGHz

# %% Define propagation fibers
def A0_func(t,T0,Ppeak0):
    return sqrt(Ppeak0)*exp(-(2*t/T0)**2)

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

def plot_power_vs_f(S,z0_vec,plot_analytical = False):     
    fig,ax = plt.subplots(constrained_layout=True)
    
    # Things for analytical gain spectrum
    beta2 = S.Fiber.beta2
    omega = S.omega
    gamma = S.Fiber.gamma[0]*8/9
    P0 = np.mean(np.abs(S.A0)**2)
    
    omega2_c = 4*gamma*P0/np.abs(beta2)
    g = np.abs(beta2*omega)*np.sqrt(omega2_c-omega**2+0j)
    ghalf = g/2
    
    omega_norm = omega/np.sqrt(omega2_c)
    K = omega_norm**2*(1-omega_norm**2)
    
    # Plotting
    idx_z_vec = []
    for z0 in z0_vec:
        idx_z_vec.append(np.argmin(np.abs(z0*1e3-S.z)))
    
    for idx_z in idx_z_vec:
        A = S.A[:,idx_z]
        AF = norm_fft(A,S.dt)
        # Normal ESD
        ESD = np.abs(AF)**2
        # Apply moving average filter
        Nav = 200
        ESD = np.convolve(np.ones(Nav),ESD,mode='same')/Nav
        PSD = hf.ESD2PSD(ESD,S.Tmax)  
        PSD_dbmghz = hf.db(PSD*1e3)
        PSD_dbmnm = PSD_dbmGHz2dbmnm(PSD_dbmghz,1550,3e8)
            
        lab = 'z='+str(int(S.z[idx_z]*1e-3))+' km'
    
        ax.plot(fftshift(S.f),PSD_dbmnm,label=lab)
        if plot_analytical:
            G = np.exp(g*S.z[idx_z])
            G = 1+2*(gamma*P0/ghalf)**2*np.sinh(ghalf*S.z[idx_z])**2
            PF_th_db = hf.db(G)+S.PSDnoise_dbmnm
            ax.plot(fftshift(S.f),fftshift(PF_th_db),color='black',ls='--')
    ax.set_xlabel('Frequency (GHz)')
    ax.set_ylabel('Power (dBm/nm)')
    ax.grid()
    ax.legend()

def lor(f,fwhm):
    return (fwhm/2)**2/((fwhm/2)**2+f**2)

def plot_Pinband_vs_z(S):
    AF = hf.norm_fft2d(S.A,S.dt)
    PF = np.abs(AF)**2
    
    f_cutoff = 200e-3
    fwhm_lor = 35e-3

    f_crop = S.f[np.abs(S.f)<f_cutoff]
    
    P_bril = np.zeros(S.Nz_save)
    P_bril_norm = np.zeros(S.Nz_save)
    for i in range(S.Nz_save):
        bril_spec = lor(f_crop,fwhm_lor)
        PF_bril = np.convolve(PF[:,i],bril_spec,mode='same')*S.df
        P_bril[i] = np.max(PF[:,i])
        P_bril_norm[i] = P_bril[i]/P_bril[0]*exp(S.Fiber.alpha*S.z[i])
    
    fig,ax = plt.subplots(2,1,constrained_layout=True)
    ax[0].plot(S.z*1e-3,P_bril_norm)
    ax[0].set_xlabel('z (km)')
    ax[0].set_ylabel('Normalized Brillouin peak power')
    ax[0].grid()
    ax[0].legend()
    ax[1].plot(S.z*1e-3,hf.db(P_bril))
    ax[1].set_xlabel('z (km)')
    ax[1].set_ylabel('Brillouin peak power(dB)')
    ax[1].grid()
    ax[1].legend()

# %% Run simulation

if __name__ == '__main__':
    # Directory for saving the data
    savedir = this_dir#+r'\data\MI_test\meas_compare_twrs\linewidth_500'

    L = 100e3               # Fiber length (m)
    T0 = 5                # Pulse length (ns)
    lam_pr = 1550e-9
    Ppeak0 = 300e-3
    dnu = 1e-9              # Linewidth of the laser (GHz)
    PSD_ase = get_ASE_noise_WGHz(G=1000,lam0=lam_pr)
    PSD_ase_dbmGHz = dbm(PSD_ase)
    PSDnoise_dbmGHz = PSD_ase_dbmGHz
    PSD_noise_dbmnm = PSD_dbmGHz2dbmnm(PSDnoise_dbmGHz,lam_pr*1e9,2.998e8)
    PSDnoise_dbmGHz = -51
    fiberdata_path = os.path.join(this_dir, 'fiber_data')

    Fiber = Passivefiber_class.from_data_sheet( fiberdata_path,
                                                'OFS_TruewaveXL.json',
                                                lam_pr)

    Tmax = T0*4             # Simulation window size (ns)
    N = 2**12
    t = np.linspace(-Tmax/2,Tmax/2,N)
    Nz_save = 51
    Nsec = 2

# %% Run simulation

if __name__ == '__main__':
    A0 = np.sqrt(Ppeak0)*np.ones(len(t))
    #A0 = A0_func(t, T0, Ppeak0)
    S = Simulation_pulsed_sections_fiber(t, A0, L, Nz_save, Fiber, Nsec,
                                        PSDnoise_dbmGHz, linewidth=dnu)
    z, A = S.run()

#S.save_pickle(savedir, 'full.pkl')
# %% Plotting 

if __name__ == '__main__':
    plt.close('all')
    plot_power_vs_tf(S,0,dbscale=True)
    plot_power_vs_tf(S,-1,dbscale=True)
    plot_Anoise(S)
    plot_power_vs_f(S,[0,5,10,15,20],plot_analytical=True)
    plot_Pinband_vs_z(S)
    plt.show()

# %%
