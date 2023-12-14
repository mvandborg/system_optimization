# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 16:40:10 2022

@author: madshv
"""
# %%

import sys
import os
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

# Insert directory of your current folder
sys.path.insert(0, 'C:/Users/madshv/OneDrive - Danmarks Tekniske Universitet/code')

import numpy as np
from scipy.constants import c,h
from numpy import pi,exp,sqrt,log10
from scipy.fft import fft,ifft,fftfreq,fftshift,ifftshift
from scipy.integrate import solve_ivp
from scipy.interpolate import splrep,splev
from scipy.optimize import fsolve
from scipy.signal import peak_widths,find_peaks
import matplotlib.pyplot as plt
from erbium_model.src.fiberdata_erbium import Erbiumfiber_class
from erbium_model.src.simulation_erbium import Erbium_simulation_class
from src.fiberdata_passive import Passivefiber_class
from src.help_functions import dbm,inv_dbm
from src.solver_system import gnls,prop_EDF
from physical_parameters import *
c = c*1e-9              # Unit m/ns

def Apr0_func(t,T0pr,Ppr0):
    H0 = t>=-T0pr/2
    H1 = t<=T0pr/2
    
    DT = T0pr/30
    H0n = t<=-(T0pr-DT)/2
    H0p = t>=-(T0pr+DT)/2
    H1n = t>=+(T0pr-DT)/2
    H1p = t<=+(T0pr+DT)/2
    H00 = t>=-(T0pr+DT)/2
    H01 = t<=+(T0pr-DT)/2
    
    #return z**0*t**0*sqrt(Ppr0)
    #return sqrt(Ppr0)*(H00*H01+H1n*H1p*(-1/DT*(t-(T0pr+DT)/2))+H0n*H0p*(1/DT*(t+(T0pr-DT)/2)))

    return sqrt(Ppr0)*exp(-(2*t/T0pr)**22)

# Physical parameters
Pp0 = 1e-6             # CW pump power (W)
Ppr0 = 250e-3            # Probe peak power (W)

# Define propagation fibers
L0 = 0.001
T0pr = 10

L_co = [150e3]
L_edf = [0.1]
L_fib = [0.1]
C = [1]    # Coupling factor

Fiber_fib0 = Fiber_SumULL
Fiber_pd0 = Fiber_SumULL

Fiber_co = [Fiber_SumULL]
Fiber_fib = [Fiber_SumULL]
Fiber_pd = [Fiber_SumULL]

Nsec = len(L_co)

# Numerical parameters
Tmax = T0pr*30          # Simulation window size (ns)

N = 2**12
dt = Tmax/N
fsam = 1/dt         # Sampling frequency (GHz)
fnyq = fsam/2

Nz = 2001
Nz_save = 101

t = np.linspace(-Tmax/2,Tmax/2,N)
f = fftfreq(N,d=dt)
df = f[1]-f[0]
omega = 2*pi*f
omega_sh = fftshift(omega)
f_sh = omega_sh/(2*pi)

# Noise floor parameters
PSDnoise_dbmHz = -121                            # Noise floor (dBm/Hz)
PSDnoise_WHz = 10**(PSDnoise_dbmHz/10)*1e-3     # Noise floor (W/Hz)
PSDnoise_WGHz = PSDnoise_WHz*1e9
ESDnoise_JGHz = PSDnoise_WGHz*Tmax              # Energy spectral density of the noise (J/Hz)

theta_noise_p = pi*np.random.uniform(size=N)
theta_noise_pr = pi*np.random.uniform(size=N)

ASDnoise_p = np.sqrt(ESDnoise_JGHz)*exp(1j*theta_noise_p)    # Amplitude spectral density (sqrt(J/Hz))
ASDnoise_pr = np.sqrt(ESDnoise_JGHz)*exp(1j*theta_noise_pr)

Anoise_p = fft(ASDnoise_p)/Tmax
Anoise_pr = fft(ASDnoise_pr)/Tmax

Ap0 = sqrt(Pp0)*np.ones(N)#+Anoise_p
Apr0 = Apr0_func(t,T0pr,Ppr0)#+Anoise_pr


# %% Fiber propagation


# Section 0 fiber (lead fiber before amplification)
A_init = np.array([Ap0,Apr0])
z_fib,Ap_fib,Apr_fib = gnls(t,omega,A_init,L0,Fiber_fib0,Nz,Nz_save)
Pp_pd_fib = Pp0*exp(-Fiber_pd0.alpha[0]*L0)
Ltot = L0

z = z_fib
Apr = Apr_fib
Ap = Ap_fib

for i in range(0,Nsec):
    # Section of co-propagation
    Ap_start = 0*Ap_fib[:,-1]+sqrt(C[i]*Pp_pd_fib)*np.ones(N)
    Apr_start = Apr_fib[:,-1]
    A_init = np.array([Ap_start,Apr_start])
    z_co,Ap_co,Apr_co = gnls(t,omega,A_init,L_co[i],Fiber_co[i],Nz,Nz_save)
    
    # Section of erbium
    Pp_cw_edf = np.max(np.abs(Ap_co[:,-1])**2)
    Ppr_cw_edf = np.max(np.abs(Apr_co[:,-1])**2)
    
    print('Pump power before edf:',Pp_cw_edf)
    z_edf,Pp_edf,Ppr_edf = prop_EDF(Fiber_edf,Nz_save,Pp_cw_edf,Ppr_cw_edf,lam_p,lam_pr,L_edf[i])
    Gpr_edf = Ppr_edf/Ppr_edf[0]
    Gp_edf = Pp_edf/Pp_edf[0]
    Ap_edf = Ap_co[:,-1]*sqrt(Gp_edf[-1])
    Apr_edf = Apr_co[:,-1]*sqrt(Gpr_edf[-1])
    
    # Section of fiber
    A_init = np.array([Ap_edf,Apr_edf])
    z_fib,Ap_fib,Apr_fib = gnls(t,omega,A_init,L_fib[i],Fiber_fib[i],Nz,Nz_save)
    
    # Section of parallel pump delivery fiber
    Lsec = L_co[i]+L_edf[i]+L_fib[i]
    Ltot = Ltot+Lsec
    Pp_pd_fib = Pp_pd_fib*(1-C[i])*exp(-Fiber_pd[i].alpha[0]*Lsec)
    
    z = np.concatenate([z,z[-1]+z_co])
    z = np.concatenate([z,z[-1]+z_fib])
    
    Ap = np.concatenate([Ap,Ap_co,Ap_fib],axis=1)
    Apr = np.concatenate([Apr,Apr_co,Apr_fib],axis=1)


# %% Post processing

AFp = fftshift(fft(Ap,axis=0),axes=0)*Tmax        # Normalized such that |AF|^2=ESD and sum(|AF|^2)*df=sum(|A|^2)*dt
AFpr = fftshift(fft(Apr,axis=0),axes=0)*Tmax

# Brillouin spectrum
dnu = 38e-3     # Spectral FWHM width (GHz)

gb_th = (dnu/2)**2/(f_sh**2+(dnu/2)**2)

Pb_th = np.zeros(AFpr.shape)
PFpr = np.abs(AFpr)**2

for i in range(Pb_th.shape[1]):
    Pb_th[1:,i] = np.convolve(gb_th[1:], PFpr[1:,i],mode='same')


# Find peak frequency

fwidth_peak = np.zeros(Pb_th.shape[1])
f_peak = np.zeros(Pb_th.shape[1])
for i in range(Pb_th.shape[1]):
    Pb_th_spl = splrep(f_sh,Pb_th[:,i])
    f_peak[i] = fsolve(lambda x:splev(x,Pb_th_spl,der=1),0)
    peaks, _ = find_peaks(Pb_th[:,i])
    fwidth_peak[i] = peak_widths(Pb_th[:,i],peaks,rel_height=0.5)[0][0]*df

# %% Save data
issave = 0

filedir = r'C:\Users\madshv\OneDrive - Danmarks Tekniske Universitet\code\data\brillouin_raman_simulation'
filename = r'test_sim.npz'

if issave==1:
    np.savez_compressed(filedir+'/'+filename,z=z,t=t,f=f,Apr=Apr,Ap=Ap,\
                        Tmax=Tmax,N=N,fsam=fsam)


# %% Plotting

PSDpr = np.abs(AFpr)**2

plt.close('all')

idxtstep = 2**1
Tmaxplot = 7*T0pr
Ntplot = int(Tmaxplot/dt)
Z,T = np.meshgrid(z,t[int((N-Ntplot)/2):int((N+Ntplot)/2):idxtstep])

Ppr_plot = np.abs(Apr[int((N-Ntplot)/2):int((N+Ntplot)/2):idxtstep])**2
Pp_plot = np.abs(Ap[int((N-Ntplot)/2):int((N+Ntplot)/2):idxtstep])**2
fig0,ax0=plt.subplots(1,2,constrained_layout=True)
im0a = ax0[0].pcolormesh(T,Z*1e-3,Pp_plot*1e3)
ax0[0].set_xlabel('Time (ns)')
ax0[0].set_ylabel('Distance (m)')
im0b = ax0[1].pcolormesh(T,Z*1e-3,Ppr_plot*1e3)
ax0[1].set_xlabel('Time (ns)')
ax0[1].set_ylabel('Distance (m)')
#cbar0a = plt.colorbar(im0a)
cbar0b = plt.colorbar(im0b)
cbar0b.set_label('Power (mW)')
ax0[0].set_title('Pump')
ax0[1].set_title('Probe')

idxfstep = 2**0
Fmaxplot = fsam/20 #5000/T0pr
Nfplot = int(Fmaxplot/df)
Z,F = np.meshgrid(z,f_sh[int((N-Nfplot)/2):int((N+Nfplot)/2):idxfstep])

AFpr_plot = dbm(np.abs(AFpr[int((N-Nfplot)/2):int((N+Nfplot)/2):idxfstep])**2)
AFp_plot = dbm(np.abs(AFp[int((N-Nfplot)/2):int((N+Nfplot)/2):idxfstep])**2)
fig1,ax1=plt.subplots(1,2,constrained_layout=True)
ax1[0].pcolormesh(F*1e3,Z*1e-3,AFp_plot)
ax1[0].set_xlabel('Frequency (MHz)')
ax1[0].set_ylabel('Distance (km)')
im1b = ax1[1].pcolormesh(F*1e3,Z*1e-3,AFpr_plot)
ax1[1].set_xlabel('Frequency (MHz)')
ax1[1].set_ylabel('Distance (km)')
cbar1b = plt.colorbar(im1b)
cbar1b.set_label('Power (dB)')
im1b.set_clim([np.max(AFp_plot),np.max(AFp_plot)-60])

fig2,ax2 = plt.subplots(2,2,constrained_layout=True)
ax2[0,0].plot(t,np.abs(Ap[:,0])**2/np.max(np.abs(Ap[:,0])**2),label='z=0')
ax2[0,0].plot(t,np.abs(Ap[:,-1])**2/np.max(np.abs(Ap[:,-1])**2),label='z=zmax')
ax2[0,1].plot(t,np.abs(Apr[:,0])**2/np.max(np.abs(Apr[:,0])**2),label='z=0')
ax2[0,1].plot(t,np.abs(Apr[:,-1])**2/np.max(np.abs(Apr[:,-1])**2),label='z=zmax')
ax2[1,0].plot(f_sh,np.abs(AFp[:,0])**2/np.max(np.abs(AFp[:,0])**2),label='z=0')
ax2[1,0].plot(f_sh,np.abs(AFp[:,-1])**2/np.max(np.abs(AFp[:,-1])**2),label='z=zmax')
ax2[1,1].plot(f_sh,np.abs(AFpr[:,0])**2/np.max(np.abs(AFpr[:,0])**2),label='z=0')
ax2[1,1].plot(f_sh,np.abs(AFpr[:,-1])**2/np.max(np.abs(AFpr[:,-1])**2),label='z=zmax')
ax2[1,0].set_xlim([-0.05,0.05])
ax2[1,1].set_xlim([-0.2,0.2])
ax2[0,0].set_xlabel('Time (ns)')
ax2[0,1].set_xlabel('Time (ns)')
ax2[1,0].set_xlabel('Frequency (GHz)')
ax2[1,1].set_xlabel('Frequency (GHz)')
ax2[0,0].set_ylabel('Power (a.u.)')
ax2[0,1].set_ylabel('Power (a.u.)')
ax2[1,0].set_ylabel('Power (a.u.)')
ax2[1,1].set_ylabel('Power (a.u.)')
ax2[0,0].legend()
ax2[0,1].legend()
ax2[1,0].legend()
ax2[1,1].legend()

fig3,ax3 = plt.subplots(constrained_layout=True)
#ax3.plot(z*1e-3,dbm(np.max(np.abs(Ap)**2,axis=0)))
#ax3.plot(z*1e-3,dbm(Pp0*exp(-alpha_p*z)),ls='--')
ax3.plot(z*1e-3,dbm(np.max(np.abs(Apr)**2,axis=0)),label='With pump')
#ax3.plot(z*1e-3,dbm(Ppr0*exp(-alpha_pr*z)),ls='--')
ax3.set_xlabel('z (m)')
ax3.set_ylabel('Power (dBm)')
ax3.legend()


fig4,ax4 = plt.subplots(1,2,constrained_layout=True)
ax4[0].plot(f_sh*1e3,np.abs(AFpr[:,104])**2/np.max(np.abs(AFpr[:,104])**2),label='23.6 km')
ax4[0].plot(f_sh*1e3,np.abs(AFpr[:,142])**2/np.max(np.abs(AFpr[:,142])**2),label='69.2 km')
ax4[0].plot(f_sh*1e3,np.abs(AFpr[:,-1])**2/np.max(np.abs(AFpr[:,-1])**2),label='300.0 km')
ax4[0].plot(f_sh*1e3,gb_th,label=r'$g(f)$',color='black',ls='--')
ax4[0].set_xlim([-100,100])
ax4[0].set_xlabel('Frequency (MHz)')
ax4[0].set_ylabel('Power (Normalized)')
ax4[0].legend()
ax4[0].set_title(r'$P_{pr}(f)$')
ax4[1].plot(f_sh*1e3,Pb_th[:,104]/np.max(Pb_th[:,104]),label='23.6 km')
ax4[1].plot(f_sh*1e3,Pb_th[:,142]/np.max(Pb_th[:,142]),label='69.2 km')
ax4[1].plot(f_sh*1e3,Pb_th[:,-1]/np.max(Pb_th[:,-1]),label='300.0 km')
ax4[1].set_xlim([-80,120])
ax4[1].set_xlabel('Frequency (MHz)')
ax4[1].set_ylabel('Power (Normalized)')
ax4[1].legend()
ax4[1].set_title(r'$g(f)*P_{pr}(f)$')

fig5,ax5 = plt.subplots(1,2,constrained_layout=True)
ax5[0].plot(z*1e-3,f_peak*1e3)
ax5[1].plot(z*1e-3,fwidth_peak*1e3)
ax5[0].set_xlabel('z (m)')
ax5[0].set_ylabel('Peak frequency (MHz)')
ax5[1].set_xlabel('z (m)')
ax5[1].set_ylabel('Brillouin spectrum width (MHz)')

fig5,ax5 = plt.subplots(constrained_layout=True)
ax5b = ax5.twinx()
ax5.plot(z*1e-3,fwidth_peak*1e3,color='tab:blue')
ax5b.plot(z*1e-3,np.max(np.abs(Apr)**2*1e3,axis=0),label='With pump',color='tab:red')
ax5.set_xlabel('z (m)')
ax5.set_ylabel('FWHM width (MHz)',color='tab:blue')
ax5b.set_ylabel('Power (mW)',color='tab:red')
ax5.set_ylim([35,55])
ax5b.set_ylim([0,230])
ax5.legend()

fig6,ax6 = plt.subplots(constrained_layout=True)
idx_tnorm = np.argmin(np.abs(t-185))
ax6.plot(t,np.abs(Apr[:,-1])**2/np.abs(Apr[idx_tnorm,-1])**2)
ax6.set_xlabel('Time (ns)')
ax6.set_ylabel('Normalized power')

plt.show()
# %%
