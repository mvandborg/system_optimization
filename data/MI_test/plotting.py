
# %% Load modules
import os
import sys
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
sys.path.insert(0, 'C:/Users/madshv/OneDrive - Danmarks Tekniske Universitet/code')
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
subfolder_path = r"C:\Users\madshv\OneDrive - Danmarks Tekniske Universitet\code\system_optimization\data\MI_test\sec2"

# List all files in the subfolder
file_list = os.listdir(subfolder_path)

class SignalAnalyzer:
    def __init__(self,subfolder_path,file_name):
        self.subfolder_path = subfolder_path
        self.file_name = file_name
    
    def extract_data(self):
        if self.file_name.endswith(".pkl"):
            file_path = os.path.join(subfolder_path, self.file_name)
            with open(file_path, "rb") as pkl_file:
                data = pickle.load(pkl_file)
                self.param = int(self.file_name.split('.')[0].split('_')[1])
                self.t = data['t']
                self.f = data['f']
                self.z = data['z']
                self.A = data['A']
                self.Fiber = dat_vec['Fiber']
                self.PSDnoise_dbmHz = dat_vec['PSDnoise_dbmHz']
                
        
def extract_data(file_list):
    # Initialize an empty list to store loaded data
    dat_vec = []
    param_sweep = []
    # Iterate through the files and load .pkl files
    for file_name in file_list:
        if file_name.endswith(".pkl"):
            file_path = os.path.join(subfolder_path, file_name)
            with open(file_path, "rb") as pkl_file:
                data = pickle.load(pkl_file)
                dat_vec.append(data)
                param_sweep.append(int(file_name.split('.')[0].split('_')[1]))
    idx_sort = np.argsort(param_sweep)
    param_sweep = np.array([param_sweep[i] for i in idx_sort])
    dat_vec = [dat_vec[i] for i in idx_sort]
    return param_sweep, dat_vec

Res = extract_data(file_list)
param_sweep = Res[0]
dat_vec = Res[1]

Nfile = len(dat_vec)


# %% Post processing
t = dat_vec[0]['t']
f = dat_vec[0]['f']
z = dat_vec[0]['z']
Fiber = dat_vec[0]['Fiber']
PSDnoise_dbmHz = dat_vec[0]['PSDnoise_dbmHz']

dt = t[1]-t[0]
Tmax = t[-1]-t[0]
df = f[1]-f[0]
N = len(t)
Nz = len(z)

T,Z = np.meshgrid(t,z)
F,Z = np.meshgrid(f,z)

SNR_vec = np.zeros([Nfile,Nz])
PSDbril_PSDmi_ratio = np.zeros([Nfile,Nz])
P_inband_vec = np.zeros([Nfile,Nz])
P_outband_vec = np.zeros([Nfile,Nz])
E_vec = np.zeros([Nfile,Nz])
z_vec = np.zeros([Nfile,Nz])
for i in range(Nfile):
    A = dat_vec[i]['A']
    L = dat_vec[i]['L']
    z = dat_vec[i]['z']
    z_vec[i] = z
    
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

    PSD_thermal = 1e-2
    PSDbril_PSDmi_ratio[i] = PSD_bril/PSD_rayscat
    SNR_vec[i] = PSD_bril/(PSD_rayscat+PSD_thermal)
    
    if isinstance(Fiber,list)==False:
        G = np.exp(Fiber.alpha[1]*L)
    else:
        G = np.exp(Fiber[0].alpha[1]*L[0]+Fiber[1].alpha[1]*L[1])
    Ltot = np.sum(L)
    fac = 1.0*(z<Ltot)+1/G*(z>Ltot)*(z<2*Ltot)+1/G**2*(z>2*Ltot)
    P_inband_vec[i] = np.sum(np.abs(AF)**2*fac*(np.abs(F.T)<0.1),axis=0)*df
    P_outband_vec[i] = np.sum(np.abs(AF)**2*fac*(np.abs(F.T)>0.1),axis=0)*df
    
    E_vec[i] = np.sum(np.abs(A)**2,axis=0)*dt

# %% Plotting
plt.close('all')
fig0,ax0 = plt.subplots(2,2,constrained_layout=True)
ax0 = ax0.flatten()
for i in range(Nfile):
    ax0[0].plot(z_vec[i]*1e-3,db(SNR_vec[i]))
ax0[0].set_ylabel('SNR (dB)')
ax0[0].set_xlabel('z (km)')
ax0[1].plot(param_sweep,db(PSDbril_PSDmi_ratio[:,-1]))
ax0[1].set_xlabel('P0 (mW)')
ax0[1].set_ylabel('PSD_bril/PSD_ray (dB)')
ax0[2].plot(param_sweep,db(SNR_vec[:,-1]))
ax0[2].set_xlabel('P0 (mW)')
ax0[2].set_ylabel('SNR (dB)')
ax0[3].plot(param_sweep,dbm(P_inband_vec[:,-2]))
ax0[3].set_xlabel(r'$P_0$')
ax0[3].set_ylabel(r'$P_{inband}$ (dBm)')

fig1,ax1 = plt.subplots(constrained_layout=True)
for i in range(Nfile):
    ax1.plot(z_vec[i]*1e-3,db(P_inband_vec[i]))


# %%
