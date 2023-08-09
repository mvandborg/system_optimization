
# %% Import modules
import sys
import os
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
sys.path.insert(0, 'C:/Users/madshv/OneDrive - Danmarks Tekniske Universitet/code')

import numpy as np
import pickle
from numpy import sqrt,exp
import matplotlib.pyplot as plt
from physical_parameters import *
from numpy.fft import fft,ifft,fftfreq,fftshift
from scipy.signal import convolve
from src.solver_system import gnls1
from src.simulation_system import Simulation_pulsed_single_fiber
from help_functions import dbm,inv_dbm,db,norm_fft,inv_norm_fft

# %% Define propagation fibers
def A0_func(t,T0,Ppeak0):
    return sqrt(Ppeak0)*exp(-(2*t/T0)**22)

L = 75e3                # Fiber length (km)
T0 = 100                # Pulse length (ns)
Ppeak0 = 150e-3         # Pump power (W)
Fiber = Fiber_SumULL
PSDnoise_dbmHz = -141   # Noise floor (dBm/Hz)

Tmax = T0*7             # Simulation window size (ns)
N = 2**16
Nz_save = 101
t = np.linspace(-Tmax/2,Tmax/2,N)
A0 = A0_func(t,T0,Ppeak0)

Sim = Simulation_pulsed_single_fiber(t,A0,L,Nz_save,Fiber,PSDnoise_dbmHz)

C_capture = 1.5e-3
C_loss = 4.5e-3
Lpulse = c/1.45*T0
C_rayscat = C_capture*C_loss*(Lpulse/2)    # Rayleigh scattering coefficient (unitless)

C_bril = 8e-9   # Brillouin scattering coefficient (unitless)

dt = Tmax/N
fsam = 1/dt         # Sampling frequency (GHz)
fnyq = fsam/2

f = fftfreq(N,d=dt)
df = f[1]-f[0]
omega = 2*pi*f
omega_sh = fftshift(omega)
f_sh = omega_sh/(2*pi)

# Noise floor parameters
PSDnoise_WHz = inv_dbm(PSDnoise_dbmHz)              # Noise floor (W/Hz)
PSDnoise_WGHz = PSDnoise_WHz*1e9                    # Noise floor (W/GHz)
ESDnoise_JGHz = PSDnoise_WGHz*Tmax                  # Energy spectral density of the noise (nJ/GHz)

# sum(|A|^2)*dt has units nJ since [A]=W and [dt]=ns
# sum(|AF|^2)*df has units nJ since [AF]=nJ/GHz and [df]=GHz
theta_noise = pi*np.random.uniform(size=N)
ASDnoise = np.sqrt(ESDnoise_JGHz)*exp(1j*theta_noise)    # Amplitude spectral density (sqrt(nJ/Hz))
Anoise = ifft(inv_norm_fft(ASDnoise,dt))                 # Normalized such that |AF|^2=ESD and sum(|AF|^2)*df=sum(|A|^2)*dt

A0 = A0_func(t,T0,Ppeak0)+Anoise
AF0 = norm_fft(fftshift(fft(A0)),dt)



# %% Run simulation
Nsec = 3

z1,A1 = gnls1(t,omega,A0,L,Fiber,Nz_save)
G = exp(Fiber.alpha[1]*L)
A0_1 = A1[:,-1]*sqrt(G)
z2,A2 = gnls1(t,omega,A0_1,L,Fiber,Nz_save)
A0_2 = A2[:,-1]*sqrt(G)
z3,A3 = gnls1(t,omega,A0_2,L,Fiber,Nz_save)  

z = np.concatenate([z1,z1[-1]+z2,z1[-1]+z2[-1]+z3])
A = np.concatenate([A1,A2,A3],axis=1)

# %% Post processing
def moving_average(a, n=3):
    ret = np.cumsum(a,axis=0, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def lor(f,f0,fwhm):
    return (fwhm/2)**2/((f-f0)**2+(fwhm/2)**2)

AF = norm_fft(fftshift(fft(A,axis=0),axes=0),dt)       # Normalized such that |AF|^2=ESD and sum(|AF|^2)*df=sum(|A|^2)*dt
PF_end = np.abs(AF[:,-1])**2
Nav = 800
PF_end_smooth = moving_average(np.abs(AF)**2,n=Nav)

# Rayleigh backscattering power
fbril = -11    # Target Brillouin frequency
idx_fbril = np.argmin(np.abs(f_sh[Nav-1:]-fbril))

# Divide by Tmax to convert from energy (ESD) to power (PSD)
ESD_rayscat = PF_end_smooth[idx_fbril]*C_rayscat    # ESD of rayleigh scattering (nJ/GHz)
PSD_rayscat = ESD_rayscat/(Tmax*1e-9)               # PSD of rayleigh scattering (W/GHz)
PSD_rayscat_dbmHz = dbm(PSD_rayscat*1e-9)

ESD_bril = np.abs(AF[int(N/2),:])**2*C_bril           # ESD of Brillouin scattering (nJ/GHz)
PSD_bril = ESD_bril/(Tmax*1e-9)                  # PSD of Brillouin scattering (W/GHz)
PSD_bril_dbmHz = dbm(PSD_bril*1e-9)

SNR = PSD_bril/PSD_rayscat

Lf = lor(f_sh,11,50e-3)
BF2 = []
for i in range(AF.shape[-1]):
    BF2.append(convolve(Lf,np.abs(AF[:,i])**2,mode='same'))
BF2 = np.array(BF2)

# %% Plotting
plt.close('all')

idx_f_jump = 2**5
idx_t_jump = 2**5

T,Z = np.meshgrid(t,z)
F,Z = np.meshgrid(f_sh,z)

yplot_t = np.abs(A[::idx_t_jump,:].T)**2
yplot_f = np.abs(AF[::idx_f_jump,:].T)**2

fig0,ax0 = plt.subplots(1,2,constrained_layout=True)
ax0[0].pcolormesh(T[:,::idx_t_jump],Z[:,::idx_t_jump]*1e-3,yplot_t)
ax0[0].set_xlabel('Time (ns)')
ax0[0].set_ylabel('Distance (km)')
ax0[1].pcolormesh(F[:,::idx_f_jump],Z[:,::idx_f_jump]*1e-3,dbm(yplot_f))
ax0[1].set_xlabel('Frequency (GHz)')
ax0[1].set_ylabel('Distance (km)')

fig1,ax1 = plt.subplots(1,2,constrained_layout=True)
ax1[0].plot(t,np.abs(A[:,-1])**2/np.max(np.abs(A[:,-1])**2))
ax1[0].plot(t,np.abs(A[:,0])**2/np.max(np.abs(A[:,0])**2))
ax1[0].set_xlabel('Time (ns)')
ax1[0].set_ylabel('Power (W)')
ax1[1].plot(f_sh,dbm(np.abs(AF[:,-1])**2/np.max(np.abs(AF[:,-1])**2)))
ax1[1].plot(f_sh,dbm(np.abs(AF[:,0])**2/np.max(np.abs(AF[:,0])**2)))
ax1[1].set_xlabel('Frequency (GHz)')
ax1[1].set_ylabel('PSD (J/GHz)')

fig2,ax2 = plt.subplots(constrained_layout=True)
ax2.plot(z,dbm(np.sum(np.abs(AF)**2,axis=0)))
ax2.plot(z,dbm(np.sum(np.abs(AF[:,0])**2,axis=0)*exp(-Fiber.alpha[1]*z)),ls='--')

# %%

fac = 1.0*(z<L)+1/G*(z>L)*(z<2*L)+1/G**2*(z>2*L)
P_inband = np.sum(np.abs(AF)**2*fac*(np.abs(F.T)<0.1),axis=0)*df
P_outband = np.sum(np.abs(AF)**2*fac*(np.abs(F.T)>0.1),axis=0)*df

fig3,ax3 = plt.subplots(1,2,constrained_layout=True)
ax3[0].plot(z*1e-3,P_inband*exp(Fiber.alpha[1]*z),label='Inband')
ax3[0].plot(z*1e-3,P_outband*exp(Fiber.alpha[1]*z),label='Outband')
ax3[0].set_ylabel('Energy')
ax3[0].set_xlabel('z (km)')
ax3[0].legend()
ax3[1].plot(z*1e-3,db(SNR))
ax3[1].set_ylabel('SNR (dB)')
ax3[1].set_xlabel('z (km)')

fig4,ax4 = plt.subplots(constrained_layout=True)
for i in [0,50,100,150,200,250,300]: 
    ax4.plot(f_sh,BF2[i]/np.max(BF2[i]))
ax4.set_xlim([-1+11,1+11])
ax4.set_xlabel('Frequency (GHz)')

# %% Save data

issave = 0
savedir = r'C:\Users\madshv\OneDrive - Danmarks Tekniske Universitet\code\system_optimization\data\MI_test\sec3\\'

if issave==1:
    savefname = 'P0_'+str(int(Ppeak0*1000))+'.pkl'
    savedict = {
        "Nsec":Nsec,
        "A":A,
        "z":z,
        "f":f_sh,
        "t":t,
        "Fiber":Fiber,
        "L":L,
        "T0":T0,
        "PSDnoise_dbmHz":PSDnoise_dbmHz,
    }
    with open(savedir+savefname, "wb") as f:
        pickle.dump(savedict, f)

# %%
