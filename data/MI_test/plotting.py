
# %% Load modules
import os
import sys
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
sys.path.insert(0, 'C:/Users/madshv/OneDrive - Danmarks Tekniske Universitet/code')
import re
import pickle
import numpy as np
from help_functions import norm_fft,moving_average,dbm,db,inv_dbm
from scipy.fft import fft,fftshift
from scipy.signal import convolve
import matplotlib.pyplot as plt
from scipy.constants import c
c = c*1e-9

# %% Import all data

# Specify the path to the subfolder containing the .pkl files
subfolder_path = r"C:\Users\madshv\OneDrive - Danmarks Tekniske Universitet\code\system_optimization\data\MI_test\sec3"

# Initialize an empty list to store loaded data
dat_vec = []

# List all files in the subfolder
file_list = os.listdir(subfolder_path)

# Iterate through the files and load .pkl files
P0 = []
for file_name in file_list:
    if file_name.endswith(".pkl"):
        file_path = os.path.join(subfolder_path, file_name)
        with open(file_path, "rb") as pkl_file:
            data = pickle.load(pkl_file)
            dat_vec.append(data)
            P0.append(int(file_name.split('.')[0].split('_')[1]))
idx_sort = np.argsort(P0)
P0 = np.array([P0[i] for i in idx_sort])
dat_vec = [dat_vec[i] for i in idx_sort]
Nfile = len(dat_vec)

# %% Post processing
z = dat_vec[0]['z']
t = dat_vec[0]['t']
f = dat_vec[0]['f']
Fiber = dat_vec[0]['Fiber']
L = dat_vec[0]['L']
PSDnoise_dbmHz = dat_vec[0]['PSDnoise_dbmHz']

dt = t[1]-t[0]
Tmax = t[-1]-t[0]
df = f[1]-f[0]
N = len(t)
Nz = len(z)

T,Z = np.meshgrid(t,z)
F,Z = np.meshgrid(f,z)

SNR_vec = np.zeros([Nfile,Nz])
P_inband_vec = np.zeros([Nfile,Nz])
P_outband_vec = np.zeros([Nfile,Nz])
for i in range(Nfile):
    A = dat_vec[i]['A']
    AF = norm_fft(fftshift(fft(A,axis=0),axes=0),dt)       # Normalized such that |AF|^2=ESD and sum(|AF|^2)*df=sum(|A|^2)*dt
    PF_end = np.abs(AF[:,-1])**2
    Nav = 800
    PF_end_smooth = moving_average(np.abs(AF)**2,n=Nav)

    # Rayleigh backscattering power
    fbril = -11    # Target Brillouin frequency
    idx_fbril = np.argmin(np.abs(f[Nav-1:]-fbril))

    # Divide by Tmax to convert from energy (ESD) to power (PSD)
    C_capture = 1.5e-3
    C_loss = 4.5e-3
    T0 = 100       # NEEDS TO BE FIXED
    Lpulse = c/1.45*T0
    C_rayscat = C_capture*C_loss*(Lpulse/2)    # Rayleigh scattering coefficient (unitless)
    C_bril = 8e-9   # Brillouin scattering coefficient (unitless)
    
    ESD_rayscat = PF_end_smooth[idx_fbril]*C_rayscat    # ESD of rayleigh scattering (nJ/GHz)
    PSD_rayscat = ESD_rayscat/(Tmax*1e-9)               # PSD of rayleigh scattering (W/GHz)
    PSD_rayscat_dbmHz = dbm(PSD_rayscat*1e-9)

    ESD_bril = np.abs(AF[int(N/2),:])**2*C_bril           # ESD of Brillouin scattering (nJ/GHz)
    PSD_bril = ESD_bril/(Tmax*1e-9)                  # PSD of Brillouin scattering (W/GHz)
    PSD_bril_dbmHz = dbm(PSD_bril*1e-9)

    SNR_vec[i] = PSD_bril/PSD_rayscat
    
    G = np.exp(Fiber.alpha[1]*L)
    fac = 1.0*(z<L)+1/G*(z>L)*(z<2*L)+1/G**2*(z>2*L)
    P_inband_vec[i] = np.sum(np.abs(AF)**2*fac*(np.abs(F.T)<0.1),axis=0)*df
    P_outband_vec[i] = np.sum(np.abs(AF)**2*fac*(np.abs(F.T)>0.1),axis=0)*df
    

# %% Plotting
plt.close('all')
fig0,ax0 = plt.subplots(1,3,constrained_layout=True)
for i in range(Nfile):
    ax0[0].plot(z*1e-3,db(SNR_vec[i]))
ax0[0].set_ylabel('SNR (dB)')
ax0[0].set_xlabel('z (km)')
ax0[1].plot(P0,db(SNR_vec[:,-1]))
ax0[1].set_xlabel('P0 (mW)')
ax0[1].set_ylabel('SNR (dB)')
ax0[2].plot(P0,db(P_inband_vec[:,-2]/P_outband_vec[:,-2]))
ax0[2].set_xlabel('P0 (mW)')
ax0[2].set_ylabel(r'$P_{inband}/P_{outband}$ (dB)')
# %%
