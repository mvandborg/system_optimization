
# %% Load modules
import os
import sys
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

import pickle
import numpy as np
from src.help_functions import norm_fft,norm_fft2d,moving_average,dbm,db,ESD2PSD
from src.help_functions import PSD_dbmGHz2dbmnm, PSD_dbmnm2dbmGHz
from scipy.fft import fft,fftshift
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt
from scipy.constants import c
c = c*1e-9  # unit m/ns

# %% Define functions

class SignalAnalyzer:
    def __init__(self,subfolder_path,file_name):
        self.subfolder_path = subfolder_path
        self.file_name = file_name
        self.extract_data()
        self.analyze_data()
        
        # Delete variables after calculations to free up RAM
        #del self.A
        #del self.Fiber_dict
    
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
        self.ESD_end_smooth = moving_average(np.abs(AF)**2,n=Nav)

        # Rayleigh backscattering power
        fbril = -10.8    # Target Brillouin frequency
        idx_fbril = np.argmin(np.abs(self.f[Nav-1:]-fbril))


        # C_loss source: https://www.fiberoptics4sale.com/blogs/archive-posts/95048006-optical-fiber-loss-and-attenuation
        lam_pr = 1550e-9    # Wavelength (m)
        lam_um = lam_pr*1e6
        K_loss = 2.0447e-4
        self.C_loss = K_loss/lam_um**4 #4.5e-5
        NA = 0.14
        n = ng = 1.46
        vg = c/ng
        self.C_capture = 1/4.2*(NA/n)**2
        self.C_rayscat = self.C_capture*self.C_loss   # Rayleigh scattering coefficient (1/m)
        
        if type(self.Fiber_dict['alpha'])==list:
            alpha_loss = self.Fiber_dict['alpha'][1]
            Aeff = self.Fiber_dict['Aeff'][0]
        else:
            alpha_loss = self.Fiber_dict['alpha']
            Aeff = self.Fiber_dict['Aeff']
        Loss_discrete = 1
        Loss = np.exp(-alpha_loss*self.z)*Loss_discrete
        
        # ESD of rayleigh scattering (nJ/GHz)
        self.PSD_rayscat = 1/2*vg*self.C_rayscat*self.ESD_end_smooth[idx_fbril]*Loss
        
        # PSD of Brillouin scattering (nJ/GHz)
        f_ac = 10.8       # Acoustic angular frequency (GHz)
        Omega = 2*np.pi*f_ac
        kB = 1.38e-14   # Unit W*ns/K
        T = 300         # Unit K
        p12=0.25        # Acoustooptic parameter (unitless)
        rho0=2202       # Density of SiO2 (kg/m3)
        eta11=3.88e-12  # Viscoelastic parameters (N*s/m2)
        fwhm = 8*np.pi*n**2*eta11/(lam_pr**2*rho0)  # FWHM of Bril. spec (GHz)
        Gamma = 2*np.pi*fwhm
        gB = 4*np.pi*n**8*p12**2/(c*1e9*lam_pr**3*rho0*f_ac*1e9*fwhm*1e9)   #2e-11 (m/W)
              # Nonlinear effectice area (m2)
        g0 = gB/Aeff    # Bril. gain coeff. (1/W/m)
        
        omega0 = 2*np.pi*c/lam_pr       # Angular frequency (2*pi*GHz)
        
        #C_bril = 8e-9   # Brillouin scattering coefficient (unitless) (from Tomin)
        self.C_bril = omega0*vg*kB*T*g0/(2*Omega)    # Bril. scat. coeff. (unitless)
        
        self.bril_scat_coeff = omega0*kB*T*Gamma*g0/(8*Omega)  # unit 1/m      
        Epr = np.sum(np.abs(self.A)**2,axis=0)*self.dt    # Pulse energy (nJ)
        self.Epr = Epr
        f_cutoff = 0.2          # Cut off frequency for filter (GHz)
        idx_filter = np.abs(self.f)<f_cutoff
        Epr_filtered = np.sum(np.abs(AF[idx_filter,:])**2,axis=0)*self.df
        self.Epr_filtered = Epr_filtered
        
        self.P_bril = vg*self.bril_scat_coeff*Epr_filtered*Loss
        self.P_bril0 = vg*self.bril_scat_coeff*Epr[0]*Loss
        self.PSD_bril = self.C_bril*Epr_filtered*Loss      # unit nJ        
        
        lor = (fwhm/2)**2/((fwhm/2)**2+self.f[idx_filter]**2)
        PSD_bril_fullspec = []
        for i in range(self.Nz):
            conv = fftconvolve(np.abs(self.AF[:,i])**2,lor,mode='same')
            PSD_bril_i = Loss[i]*self.C_bril*conv*self.df
            PSD_bril_fullspec.append(PSD_bril_i)
        self.PSD_bril_fullspec = np.array(PSD_bril_fullspec).T
        
        q = 1.602e-19           # Fundamental electron charge (C) 
        Rp = 1.05               # Responsivity (A/W)
        B = 0.1                 # Bandwidth of photodiode (GHz)
        NF_db = 8               # Noise figure of LNA
        NF = 10**(NF_db/10)
        
        self.PSD_shot = NF*q/Rp*1e9         #q/(2*Rp)*1e9    # Convert from J to nJ
        self.P_shot = self.PSD_shot*B       # Shot noise power (W)
        self.P_rayscat = self.PSD_rayscat*B # Rayleigh scattering power (W)
        
        self.ER_db = 37.75#45
        ER = 10**(self.ER_db/10)
        self.Ppr0 = np.max(np.abs(self.A[:,0])**2)
        self.Leff = (1-np.exp(-2*alpha_loss*self.L))/(2*alpha_loss)
        self.P_b2 = self.bril_scat_coeff*self.Ppr0/ER*self.Leff
        
        # Squaring in envelope detection
        #self.PSD_noise = np.sqrt(self.PSD_shot**2+
        #                    self.PSD_rayscat**2+
        #                    self.PSD_bril**2+
        #                    4*self.PSD_shot*self.PSD_rayscat+
        #                    4*self.PSD_shot*self.PSD_bril+
        #                    (4**2-np.pi**2)/4*self.PSD_rayscat*self.PSD_bril
        #)
        #self.PSD_signal = ( self.PSD_bril+
        #                    self.PSD_rayscat+
        #                    0*self.PSD_shot+
        #                    np.pi/2*np.sqrt(self.PSD_bril*self.PSD_rayscat)
        #)
        
        # Without squaring in envelope detection
        self.PSD_noise = self.PSD_shot+self.PSD_rayscat+self.PSD_bril
        self.PSD_signal = self.PSD_bril
        self.P_noise = self.P_bril+self.P_shot+self.P_rayscat+self.P_b2
        self.P_signal = self.P_bril
        
        print(  'Psignal={:.3f}'.format(self.P_signal[0]*1e9),
                'Pshot={:.3f}'.format(self.P_shot*1e9),
                'Pray={:.3f}'.format(self.P_rayscat[0]*1e9),
                'Pb2={:.3f}'.format(self.P_b2[0]*1e9))
        
        self.SNR = 1.75*self.P_signal/self.P_noise
        
        if isinstance(self.Fiber_dict,list)==True:
            if isinstance(self.L[0],list)==True:
                G = 1
            else:
                G = np.exp(self.Fiber_dict[0]['alpha'][1]*self.L[0]+\
                    self.Fiber_dict[1]['alpha'][1]*self.L[1])
        else:
            G = np.exp(alpha_loss*self.L)
        Ltot = np.sum(np.sum(self.L))
        
        self.fac = 1.0*(self.z<Ltot)+1/G*(self.z>Ltot)*(self.z<2*Ltot)+1/G**2*(self.z>2*Ltot)
        F = np.tile(self.f, (self.Nz, 1))
        self.E_inband = np.sum(np.abs(AF)**2*(np.abs(F.T)<0.1),axis=0)*self.df
        self.E_outband = np.sum(np.abs(AF)**2*(np.abs(F.T)>0.1),axis=0)*self.df
        
        self.E = np.sum(np.abs(self.A)**2,axis=0)*self.dt
    
def sort_res(param_vec,Res_vec):
    idx_param_vec = np.argsort(param_vec)
    param_vec = [param_vec[i] for i in idx_param_vec]
    Res_vec = [Res_vec[i] for i in idx_param_vec]
    return param_vec,Res_vec


def plot_Pinband_vs_z(R):
    fig,ax = plt.subplots(constrained_layout=True)
    for i in range(len(R)):
        ax.plot(R[i].z*1e-3,db(R[i].E_inband),label=R[i].param)
    ax.set_xlabel('z (km)')
    ax.set_ylabel('Inband energy (dB)')
    ax.legend()
    ax.grid()

def plot_Pbril_vs_z(R):
    fig,ax = plt.subplots(constrained_layout=True)
    R = [R[i] for i in [0,2,4,6,8,10,12]]
    for i in range(len(R)):
        ax.plot(R[i].z*1e-3,dbm(R[i].P_bril),label=R[i].param)
    ax.set_xlabel('z (km)')
    ax.set_ylabel('Brillouin peak power (dBm)')
    ax.legend()
    ax.grid()

def plot_PSD_vs_lam(R,z0=0):
    idx_z = np.argmin(np.abs(R[0].z-z0))
    fig,ax = plt.subplots(constrained_layout=True)
    for i in range(len(R)):
        AF = norm_fft(R[i].A[:,idx_z],R[i].dt)
        ESD = np.abs(AF)**2
        ESD = ESD2PSD(ESD,R[i].Tmax)
        Nconv = 1
        ESD_smooth = np.convolve(ESD,np.ones(Nconv),mode='same')/Nconv
        ax.plot(R[i].f,PSD_dbmGHz2dbmnm(dbm(ESD_smooth),1550,3e8),
                label=str(R[i].param))
        ax.set_xlabel('Frequency (GHz)')
        ax.set_ylabel('PSD (dBm/nm) @ '+"{:.1f}".format(R[0].z[idx_z]*1e-3)+' km')
    ax.legend()
    ax.grid()

def plot_PSDbril_PSDmi_ratio_vs_z(R):
    fig,ax = plt.subplots(constrained_layout=True)
    for i in range(Nfile):
        ax.plot(R[i].z*1e-3,db(R[i].PSD_bril/R[i].PSD_rayscat))
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

def plot_PSDbril_vs_sweepparam(param_vec,R,zi):
    idxz = np.argmin(np.abs(zi-R[0].z))
    fig,ax = plt.subplots(constrained_layout=True)
    y_bril = dbm([r.PSD_bril[idxz] for r in R])
    y_shot = dbm([r.PSD_shot for r in R])
    y_ray = dbm([r.PSD_rayscat[idxz] for r in R])
    ax.plot(param_vec,y_bril,label=r'$PSD_{bril}$')
    ax.plot(param_vec,y_ray,label=r'$PSD_{rayscat}$')
    ax.plot(param_vec,y_shot,label=r'$PSD_{shot}$')
    ax.set_xlabel(r'Sweep parameter')
    ax.set_ylabel(r'$PSD$ (dBm/GHz) @ '+"{:.1f}".format(R[0].z[idxz]*1e-3)+' km')
    ax.grid()
    ax.legend()

def plot_SNR_vs_z(param_vec,R,Navg=1):
    SNR_max = 2/np.pi/(1-2/np.pi)*Navg
    fig,ax = plt.subplots(constrained_layout=True)
    for r in R:
        ax.plot(r.z*1e-3,
                db(r.SNR*Navg),
                label=r.param)
    ax.plot([r.z[0]*1e-3,r.z[-1]*1e-3],
            [db(SNR_max),db(SNR_max)],
            label='Max',
            color='black',
            ls='--')
    ax.set_xlabel(r'z (km)')
    ax.set_ylabel('SNR (dB)')
    ax.grid()
    ax.legend()
    
def plot_PSDnoise_vs_z(param_vec,R):
    fig,ax = plt.subplots(constrained_layout=True)
    for r in R:
        ax.plot(r.z*1e-3,dbm(r.P_noise),label=r.param)
    ax.set_xlabel(r'z (km)')
    ax.set_ylabel('Noise power (dBm)')
    ax.grid()
    ax.legend()

def plot_SNR_vs_sweepparam(param_vec,R,z0=0,dBscale=False,Navg=1):
    idx_z = np.argmin(np.abs(R[0].z-z0))
    fig,ax = plt.subplots(constrained_layout=True)
    y = np.array([r.SNR[idx_z] for r in R])*Navg
    labeltext = 'SNR @ '
    if dBscale:
        y = db([r.SNR[idx_z] for r in R])
        labeltext = 'SNR (dB) @ '
    ax.plot(param_vec,y)
    ax.set_xlabel(r'Sweep parameter')
    ax.set_ylabel(labeltext+"{:.1f}".format(R[0].z[idx_z]*1e-3)+' km')
    ax.grid()

def plot_PSD_at_start(R):
    plot_PSD_at_zi(R,0)
    
def plot_PSD_at_zi(R0,zi):
    idx_z = np.argmin(np.abs(R0.z-zi))
    ESD = np.abs(R0.AF[:,idx_z])**2
    Tmax = R0.t[-1]-R0.t[0]
    PSD = ESD2PSD(ESD,Tmax)
    
    fig,ax = plt.subplots(constrained_layout=True)
    ax.plot(R0.f,PSD_dbmGHz2dbmnm(dbm(PSD),1550,3e8))
    ax.set_xlabel(r'Frequency (GHz)')
    ax.set_ylabel(r'PSD (dBm/nm) @'+"{:.1f}".format(R[0].z[idx_z]*1e-3)+' km')
    ax.grid()

def plot_PSD_bril_at_zi(R0,zi):
    idx_z = np.argmin(np.abs(R0.z-zi))
    fig,ax = plt.subplots(constrained_layout=True)
    ax.plot(R0.f,R0.PSD_bril_fullspec[:,idx_z])
    ax.set_xlabel(r'Frequency (GHz)')
    ax.set_ylabel(r'PSD(dBm/GHz) @ '+"{:.1f}".format(R[0].z[idx_z]*1e-3)+' km')
    ax.grid()

# %% Rune code
if __name__ == '__main__':
    # Specify the path to the subfolder containing the .pkl files
    subfolder_path = file_dir+r"\data\MI_test\meas_compare_dfm\noise_-15_fitpower"

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

# %% Plotting
if __name__ == '__main__':
    z0 = 40e3
    Navg = 1024
    plt.close('all')
    #plot_Pinband_vs_z(R)
    plot_Pbril_vs_z(R)
    plot_PSD_vs_lam(R,z0=z0)
    #plot_PSDbril_PSDmi_ratio_vs_z(R)
    plot_PSDnoise_vs_z(param_vec,R)
    plot_SNR_vs_z(param_vec,R,Navg=Navg)
    #plot_PSDbril_PSDmi_ratio_vs_sweepparam(param_vec,R)
    plot_PSDbril_vs_sweepparam(param_vec,R,z0)
    plot_SNR_vs_sweepparam(param_vec,R,z0=z0,Navg=Navg)
    plot_PSD_at_zi(R[-1],0)
    plt.show()

# %%
