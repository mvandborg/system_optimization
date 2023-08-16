
# %% Load modules
import os
import sys
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
sys.path.insert(0, 'C:/Users/madshv/OneDrive - Danmarks Tekniske Universitet/code')
import pickle
import numpy as np
from help_functions import norm_fft,norm_fft2d,moving_average,dbm,db,ESD2PSD
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
        del self.Fiber
    
    def extract_data(self):
        if self.file_name.endswith(".pkl"):
            file_path = os.path.join(self.subfolder_path, self.file_name)
            with open(file_path, "rb") as pkl_file:
                data = pickle.load(pkl_file)
                self.t = data['t']
                self.f = data['f']
                self.z = data['z']
                self.A = data['A']
                self.Fiber = data['Fiber']
                self.PSDnoise_dbmHz = data['PSDnoise_dbmHz']
                self.L = data['L']
                self.Nz = len(self.z)
                self.dt = self.t[1]-self.t[0]
                self.Tmax = self.t[-1]-self.t[0]
                self.df = self.f[1]-self.f[0]
                self.N = len(self.t)
                
                self.param = int(self.file_name.split('.')[0].split('_')[1])
    
    def analyze_data(self):
        # fft: Normalized such that |AF|^2=ESD and sum(|AF|^2)*df=sum(|A|^2)*dt
        AF = norm_fft2d(self.A,self.dt,axis=0)
        Nav = 800
        self.PF_end_smooth = moving_average(np.abs(AF)**2,n=Nav)

        # Rayleigh backscattering power
        fbril = -11    # Target Brillouin frequency
        idx_fbril = np.argmin(np.abs(self.f[Nav-1:]-fbril))

        # Divide by Tmax to convert from energy (ESD) to power (PSD)
        C_capture = 1.5e-3
        C_loss = 4.5e-3
        T0 = 100       # NEEDS TO BE FIXED
        Lpulse = c/1.45*T0
        C_rayscat = C_capture*C_loss*(Lpulse/2)    # Rayleigh scattering coefficient (unitless)
        C_bril = 8e-9   # Brillouin scattering coefficient (unitless)
                
        self.ESD_rayscat = self.PF_end_smooth[idx_fbril]*C_rayscat              # ESD of rayleigh scattering (nJ/GHz)
        self.PSD_rayscat = ESD2PSD(self.ESD_rayscat,self.Tmax)                  # PSD of rayleigh scattering (W/GHz)

        self.ESD_bril = np.abs(AF[int(self.N/2),:])**2*C_bril                   # ESD of Brillouin scattering (nJ/GHz)
        self.PSD_bril = ESD2PSD(self.ESD_bril,self.Tmax)                        # PSD of Brillouin scattering (W/GHz)

        self.PSDbril_PSDmi_ratio = self.PSD_bril/self.PSD_rayscat
        
        if isinstance(self.Fiber,list)==False:
            G = np.exp(self.Fiber.alpha[1]*self.L)
        else:
            G = np.exp(self.Fiber[0].alpha[1]*self.L[0]+self.Fiber[1].alpha[1]*self.L[1])
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

# Plotting
def plotting():
    plt.close('all')
    fig0,ax0 = plt.subplots(2,2,constrained_layout=True)
    ax0 = ax0.flatten()
    for i in range(Nfile):
        ax0[0].plot(R[i].z*1e-3,db(R[i].PSDbril_PSDmi_ratio))
    ax0[0].set_ylabel(r'$PSD_{bril}/PSD_{ray}$ (dB)')
    ax0[0].set_xlabel('z (km)')

    y = db([r.PSDbril_PSDmi_ratio[-1] for r in R])
    ax0[1].plot(param_vec,y)
    #ax0[1].set_xlabel('P0 (mW)')
    ax0[1].set_xlabel(r'$PSD_{noise}$ (dBm/Hz)')
    ax0[1].set_ylabel(r'$PSD_{bril}/PSD_{ray}$ (dB)')
    ax0[1].grid()
    y = dbm([r.P_inband[-2] for r in R])
    ax0[3].plot(param_vec,y)
    #ax0[3].set_xlabel(r'$P_0$ (mW)')
    ax0[3].set_xlabel(r'$PSD_{noise}$ (dBm/Hz)')
    ax0[3].set_ylabel(r'$E_{inband}$ (dB)')
    ax0[3].grid()

    fig1,ax1 = plt.subplots(constrained_layout=True)
    for i in range(Nfile):
        ax1.plot(R[i].z*1e-3,db(R[i].P_inband))
    
    AF_end = norm_fft(R[0].A[:,0],R[0].dt)
    ESD_end = np.abs(AF_end)**2
    fig2,ax2 = plt.subplots(constrained_layout=True)
    ax2.plot(R[0].f,dbm(ESD_end))
    ax2.set_xlabel('Frequency (GHz)')
    ax2.set_ylabel('PSD')

# %% Rune code

if __name__ == '__main__':
    # Specify the path to the subfolder containing the .pkl files
    subfolder_path = r"C:\Users\madshv\OneDrive - Danmarks Tekniske Universitet\code\system_optimization\data\MI_test\noise_level_sweep"

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
    
    plotting()



# %%
