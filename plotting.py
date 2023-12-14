
# %% Load modules
import os
import sys
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

import pickle
import numpy as np
from .src.help_functions import norm_fft,norm_fft2d,moving_average,dbm,db,ESD2PSD
from scipy.fft import fft,fftshift
import matplotlib.pyplot as plt
from scipy.constants import c
c = c*1e-9

# %% Define functions

class SignalAnalyzer:
    def __init__(self,subfolder_path,file_name):
        self.subfolder_path = subfolder_path
        self.file_name = file_name
        self.extract_data()
        self.analyze_data()
        
        # Delete variables after calculations to free up RAM
        #del self.A
        del self.Fiber_dict
    
    def extract_data(self):
        if self.file_name.endswith(".pkl"):
            file_path = os.path.join(self.subfolder_path, self.file_name)
            with open(file_path, "rb") as pkl_file:
                data = pickle.load(pkl_file)
                self.Fiber_dict = data['Fiber_dict']
                self.t = data['t']
                self.f = data['f']
                self.z = data['z']
                self.A = data['A']
                self.PSDnoise_dbmHz = data['PSDnoise_dbmHz']
                self.L = data['L']
                self.PSDnoise_dbmGHz = self.PSDnoise_dbmHz+90
                self.Nz = len(self.z)
                self.dt = self.t[1]-self.t[0]
                self.Tmax = self.t[-1]-self.t[0]
                self.df = self.f[1]-self.f[0]
                self.N = len(self.t)
                
                self.param = int(self.file_name.split('.')[0].split('_')[1])
    
    def analyze_data(self):
        # fft: Normalized such that |AF|^2=ESD and sum(|AF|^2)*df=sum(|A|^2)*dt
        AF = norm_fft2d(self.A,self.dt,axis=0)
        self.AF = AF
        Nav = 800
        self.PF_end_smooth = moving_average(np.abs(AF)**2,n=Nav)

        # Rayleigh backscattering power
        fbril = -11    # Target Brillouin frequency
        idx_fbril = np.argmin(np.abs(self.f[Nav-1:]-fbril))

        C_capture = 1.5e-3
        C_loss = 4.5e-3
        T0 = 100       # NEEDS TO BE FIXED
        Lpulse = c/1.45*T0
        C_rayscat = C_capture*C_loss*(Lpulse/2)    # Rayleigh scattering coefficient (unitless)
        C_bril = 8e-9   # Brillouin scattering coefficient (unitless)
        
        self.C_rayscat = C_rayscat
        self.C_bril = C_bril
        
        # ESD of rayleigh scattering (nJ/GHz)
        self.ESD_rayscat = self.PF_end_smooth[idx_fbril]*C_rayscat
        # PSD of rayleigh scattering (W/GHz)              
        self.PSD_rayscat = ESD2PSD(self.ESD_rayscat,self.Tmax)
                       
        # ESD of Brillouin scattering (nJ/GHz)
        self.ESD_bril = np.abs(AF[int(self.N/2),:])**2*C_bril
        # PSD of Brillouin scattering (W/GHz)                   
        self.PSD_bril = ESD2PSD(self.ESD_bril,self.Tmax)                        

        self.PSDbril_PSDmi_ratio = self.PSD_bril/self.PSD_rayscat
        q = 1.602e-19       # Fundamental electron charge (C) 
        Rp = 0.8            # Responsivity (A/W)
        B = q/(2*Rp)*1e9    # Convert from J to nJ
        
        self.SNR = 1/(1/self.PSDbril_PSDmi_ratio+B/self.PSD_bril)
        
        if isinstance(self.Fiber_dict,list)==False:
            G = np.exp(self.Fiber_dict['alpha'][1]*self.L)
        else:
            G = np.exp(self.Fiber_dict[0]['alpha'][1]*self.L[0]+self.Fiber_dict[1]['alpha'][1]*self.L[1])
        Ltot = np.sum(self.L)
        self.fac = 1.0*(self.z<Ltot)+1/G*(self.z>Ltot)*(self.z<2*Ltot)+1/G**2*(self.z>2*Ltot)
        F = np.tile(self.f, (self.Nz, 1))
        self.P_inband = np.sum(np.abs(AF)**2*(np.abs(F.T)<0.1),axis=0)*self.df
        self.P_outband = np.sum(np.abs(AF)**2*(np.abs(F.T)>0.1),axis=0)*self.df
        
        self.E = np.sum(np.abs(self.A)**2,axis=0)*self.dt
    
def sort_res(param_vec,Res_vec):
    idx_param_vec = np.argsort(param_vec)
    param_vec = [param_vec[i] for i in idx_param_vec]
    Res_vec = [Res_vec[i] for i in idx_param_vec]
    return param_vec,Res_vec


def plot_Pinband_vs_z(R):
    fig,ax = plt.subplots(constrained_layout=True)
    for i in range(len(R)):
        ax.plot(R[i].z*1e-3,dbm(R[i].P_inband))
    ax.set_xlabel('z (km)')
    ax.set_ylabel('Power (dBm)')
    ax.grid()

def plot_PSD_vs_lam(R):
    fig,ax = plt.subplots(constrained_layout=True)
    for i in range(len(R)):
        AF_end = norm_fft(R[i].A[:,-1],R[i].dt)
        ESD_end = np.abs(AF_end)**2
        PSD_end = ESD2PSD(ESD_end,R[i].Tmax)
        ax.plot(R[i].f*1550**2/3e8+1550,dbm(PSD_end)+i*100,
                label=f'Noise={R[i].PSDnoise_dbmGHz} dBm/GHz')
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('PSD (dBm/GHz)')
    ax.legend()
    ax.grid()

def plot_PSDbril_PSDmi_ratio_vs_z(R):
    fig,ax = plt.subplots(constrained_layout=True)
    for i in range(Nfile):
        ax.plot(R[i].z*1e-3,db(R[i].PSDbril_PSDmi_ratio))
    ax.set_ylabel(r'$PSD_{bril}/PSD_{ray}$ (dB)')
    ax.set_xlabel('z (km)')
    ax.grid()

def plot_PSDbril_PSDmi_ratio_vs_sweepparam(param_vec,R):
    fig,ax = plt.subplots(constrained_layout=True)
    y = db([r.PSDbril_PSDmi_ratio[-1] for r in R])
    ax.plot(param_vec,y)
    ax.set_xlabel(r'Sweep parameter')
    ax.set_ylabel(r'$PSD_{bril}/PSD_{ray}$ (dB)')
    ax.grid()

def plot_PSDbril_vs_sweepparam(param_vec,R):
    fig,ax = plt.subplots(constrained_layout=True)
    y = dbm([r.PSD_bril[-2] for r in R])
    ax.plot(param_vec,y)
    ax.set_xlabel(r'Sweep parameter')
    ax.set_ylabel(r'$PSD_{bril}$ (dBm/GHz)')
    ax.grid()

def plot_SNR_vs_sweepparam(param_vec,R):
    fig,ax = plt.subplots(constrained_layout=True)
    y = db([r.SNR[-2] for r in R])
    ax.plot(param_vec,y)
    ax.set_xlabel(r'Sweep parameter')
    ax.set_ylabel(r'$SNR$ (dB)')
    ax.grid()

def plot_PSD_at_start(R):
    idx_plot = 0
    R0 = R[idx_plot]
    fig,ax = plt.subplots(constrained_layout=True)
    ESD = np.abs(R0.AF[:,0])**2
    Tmax = R0.t[-1]-R0.t[0]
    PSD = ESD2PSD(ESD,Tmax)
    ax.plot(R0.f,dbm(PSD))
    ax.set_xlabel(r'Frequency (GHz)')
    ax.set_ylabel(r'PSD @ z=0 (dBm/GHz)')
    ax.grid()

# %% Rune code

if __name__ == '__main__':
    # Specify the path to the subfolder containing the .pkl files
    subfolder_path = r"C:\Users\madshv\OneDrive - Danmarks Tekniske Universitet\code\system_optimization\data\MI_test\altfiber_sec3\P150"

    # List all files in the subfolder
    file_list = os.listdir(subfolder_path)
    Nfile = len(file_list)
    R = []
    param_vec = []

    for file_name in file_list:
        R1 = SignalAnalyzer(subfolder_path,file_name)
        R.append(R1)
        param_vec.append(R1.param)

    param_vec,R = sort_res(param_vec,R)
    
    plt.close('all')
    plot_Pinband_vs_z(R)
    plot_PSD_vs_lam(R)
    plot_PSDbril_PSDmi_ratio_vs_z(R)
    plot_PSDbril_PSDmi_ratio_vs_sweepparam(param_vec,R)
    plot_PSDbril_vs_sweepparam(param_vec,R)
    plot_SNR_vs_sweepparam(param_vec,R)
    plot_PSD_at_start(R)
    plt.show()

# %%
