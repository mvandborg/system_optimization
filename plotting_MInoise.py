
# %% Load modules
import os
import sys
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

import pickle
import numpy as np
from src.help_functions import norm_fft,norm_fft2d,moving_average,dbm,db,\
    ESD2PSD,PSD_dbmGHz2dbmnm, PSD_dbmnm2dbmGHz
from scipy.fft import fft,fftshift
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt
from scipy.constants import c
from numpy import pi
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

        # Puls energy calculations
        Epr = np.sum(np.abs(self.A)**2,axis=0)*self.dt    # Pulse energy (nJ)
        self.Epr = Epr
        f_cutoff = 0.1          # Cut off frequency for filter (GHz)
        idx_filter = np.abs(self.f)<f_cutoff
        self.Epr = np.sum(np.abs(self.A)**2,axis=0)*self.dt
        self.Epr_filtered = np.sum(np.abs(AF[idx_filter,:])**2,axis=0)*self.df
        
        # Initial pulse power
        self.Ppr0 = np.mean(np.abs(self.A[:,0])**2) # When using CW simulation
        #self.Ppr0 = np.max(np.abs(self.A[:,0])**2) # When using pulsed sim
        
        self.Ppr = self.Epr_filtered/self.Epr_filtered[0]*self.Ppr0
        
        # Physical constants
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
        # Rayleigh scattering coefficient (1/m)
        self.C_rayscat = self.C_capture*self.C_loss   
        
        # PSD of Brillouin scattering (nJ/GHz)
        f_ac = 10.8       # Acoustic angular frequency (GHz)
        Omega = 2*pi*f_ac
        kB = 1.38e-14   # Unit W*ns/K
        T = 300         # Unit K
        p12=0.25        # Acoustooptic parameter (unitless)
        rho0=2202       # Density of SiO2 (kg/m3)
        eta11=3.88e-12  # Viscoelastic parameters (N*s/m2)
        fwhm = 8*pi*n**2*eta11/(lam_pr**2*rho0)  # FWHM of Bril. spec (GHz)
        Gamma = 2*pi*fwhm
        gB = 4*pi*n**8*p12**2/(c*1e9*lam_pr**3*rho0*f_ac*1e9*fwhm*1e9)   #2e-11 (m/W) from Tomin's thesis
        omega0 = 2*pi*c/lam_pr       # Angular frequency (2*pi*GHz)
        
        # lorentzian spectrum
        lor = (fwhm/2)**2/((fwhm/2)**2+self.f[idx_filter]**2)
        
        # Calculate the loss for different simulation configurations
        # If two fiber types have been used
        if type(self.Fiber_dict)==list:
            alpha_loss = np.array([f['alpha'] for f in self.Fiber_dict])
            Aeff = np.array([f['alpha'] for f in self.Fiber_dict])
            z_sec1 = np.clip(self.z,a_min=0,a_max=self.L[0])
            z_sec2 = np.clip(self.z,a_min=self.L[0],a_max=None)
            Loss1 = np.exp(-alpha_loss[0]*z_sec1)
            Loss2 = np.exp(-alpha_loss[1]*(z_sec2-self.L[0]))
            Loss = Loss1*Loss2
            Loss_sec = np.exp(-alpha_loss[0]*self.L[0]-alpha_loss[1]*self.L[1])
            g0 = np.ones(len(Loss))
            g0[self.z<self.L[0]] = gB/Aeff[0]
            g0[self.z>self.L[0]] = gB/Aeff[1]
            
            # Leff of two sections of fiber
            Leff1 = (1-Loss1**2)/(2*alpha_loss[0])
            Loss2a = np.exp(-alpha_loss[1]*self.L[0])
            Leff2 = Loss2a**2*(1-Loss2**2)/(2*alpha_loss[1])
            self.Leff = Leff1+Leff2
            
        else:
            # If single fiber, but more than one wavelength
            if isinstance(self.Fiber_dict['alpha'],(list,np.ndarray)):
                alpha_loss = self.Fiber_dict['alpha'][1]
                Aeff = self.Fiber_dict['Aeff'][0]
            else:
                # If single fiber and one wavelength
                alpha_loss = self.Fiber_dict['alpha']
                Aeff = self.Fiber_dict['Aeff'][0]
            Loss = np.exp(-alpha_loss*self.z)
            Loss_sec = np.exp(-alpha_loss*self.L)
            g0 =gB/Aeff*np.ones(len(Loss))  # g0 at the probe wavelength
            self.Leff = (1-Loss**2)/alpha_loss
        self.Loss = Loss
        
        # Apply gain between each section
        Ltot = np.sum(np.sum(self.L))
        G = 1/Loss_sec
        
        self.Loss[self.z>Ltot]*=G
        self.Loss[self.z>2*Ltot]*=G
        
        # ESD of rayleigh scattering (nJ/GHz)
        self.PSD_rayscat = 1/2*vg*self.C_rayscat*\
            self.ESD_end_smooth[idx_fbril]*Loss
        B = 0.1 # Bandwidth of photo-detector (GHz)
        self.P_rayscat = self.PSD_rayscat*B 
             
        #C_bril = 8e-9   # Brillouin scattering coefficient (unitless) (from Tomin)
        # Brillouin scattering coefficient (unit unit 1/m)
        #  (Pb=C_b*Lpulse/2*Ppr)
        self.C_bril = omega0*kB*T*Gamma*g0/(4*Omega)
        # Brillouin scattering coefficient (unit 1/s)
        # unit 1/m (Pb=C_b*Lpulse/2*Epr)
        self.C_bril_energy = 1/2*vg*self.C_bril
        # Brillouin scattering coefficient (unit 1/s)
        # unit 1/m (Pb=C_b*Lpulse/2*Epr)
        self.C_bril_psd = omega0*kB*T*Gamma*vg*g0/(8*pi*Omega)
        
        self.P_bril = self.C_bril_energy*self.Epr_filtered*Loss
        self.PSD_bril = self.C_bril_psd*self.Epr_filtered*Loss      # unit nJ        
        
        PSD_bril_fullspec = []
        for i in range(self.Nz):
            conv = fftconvolve(np.abs(self.AF[:,i])**2,lor,mode='same')
            PSD_bril_i = Loss[i]*self.C_bril_psd[i]*conv*self.df
            PSD_bril_fullspec.append(PSD_bril_i)
        self.PSD_bril_fullspec = np.array(PSD_bril_fullspec).T
                
        # Manual from fit (no power fitting):
        fit_dfm = {
            'A0':13.802*1e3,
            'A1':1.1005*1e3,
            'A2':1609.33,
            'Lpulse_half':37.5
        }
        fit_lios = {
            'A0':6.37e-2*1e3,
            'A1':0,
            'A2':0,
            'A0_noise':1.13e-3*1e3,
            'A1_noise':0,
            'A2_noise':4.24e-3,
            'Lpulse_half':10
        }
        fit = fit_dfm
        
        if fit==fit_dfm:
            Nens_avg = 2**10
            k2_N = 1.75*Nens_avg
            self.SNR_max = (k2_N+1)/np.sqrt(2*(2*k2_N+1))
            
            self.k = pi*fit['A0']/(self.C_bril[0]*fit['Lpulse_half'])
            self.NEP = fit['A2']/self.k 
            self.ER = fit['A1']/self.k*self.C_bril[0]*self.Leff[-1]
            
            self.P_bril = self.C_bril[0]*fit['Lpulse_half']*Loss*self.Ppr
            self.P_b2 = self.C_bril[0]*self.Ppr0*self.ER*self.Leff[-1]
            
            # Without squaring in envelope detection
            self.P_noise = self.P_bril+self.NEP+self.P_rayscat+self.P_b2
            self.P_signal = self.P_bril
            self.SNR = self.SNR_max*self.P_signal/self.P_noise
            # Fitting
            self.V2_signal = fit['A0']*self.Ppr*Loss
            self.V2_noise = (self.V2_signal+fit['A1']*self.Ppr0+\
                fit['A2'])
        elif fit==fit_lios:
            Nens_avg = 1000
            
            # <v>=A0*Ppr*Loss^2
            # std(v)=A0_noise*Ppr*Loss^2+A1_noise*Ppr+A2_noise
            
            self.SNR_max = fit['A0']/fit['A0_noise']
            
            self.k = pi*fit['A0']/(self.C_bril[0]*fit['Lpulse_half'])
            self.k_n = pi*fit['A0_noise']/(self.C_bril[0]*fit['Lpulse_half'])
            
            self.NEP = fit['A2_noise']/self.k_n 
            self.ER = fit['A1_noise']*self.C_bril[0]*self.Leff[-1]/self.k
            
            self.P_bril = self.C_bril[0]*fit['Lpulse_half']*Loss*self.Ppr
            self.P_b2 = self.C_bril[0]*self.Ppr0*self.ER*self.Leff[-1]
            
            # Without squaring in envelope detection
            self.P_noise = self.P_bril+self.NEP+self.P_rayscat+self.P_b2
            self.P_signal = self.P_bril
            
            # Fitting
            self.P_signal = fit['A0']*self.Ppr*Loss
            self.P_noise = fit['A0_noise']*self.Ppr*Loss+fit['A2_noise']
            
            self.SNR = self.P_signal/self.P_noise
                
    
def sort_res(param_vec,Res_vec):
    idx_param_vec = np.argsort(param_vec)
    param_vec = [param_vec[i] for i in idx_param_vec]
    Res_vec = [Res_vec[i] for i in idx_param_vec]
    return param_vec,Res_vec


def plot_Pinband_vs_z(R,plot_norm=False):
    fig,ax = plt.subplots(constrained_layout=True,figsize=(5,4))
    if plot_norm==False:
        for i in range(len(R)):
            ax.plot(R[i].z*1e-3,db(R[i].Epr_filtered),label=R[i].param)
        ax.set_ylabel('Inband energy (dB)')
    else:
        for i in range(len(R)):
            Loss = R[0].Loss
            PSD_dbmnm = PSD_dbmGHz2dbmnm(R[i].PSDnoise_dbmGHz,1550,3e8)
            ax.plot(R[i].z*1e-3,
                    R[i].Epr_filtered/R[i].Epr_filtered[0]/Loss,
                    label="{:.0f} dBm/nm".format(PSD_dbmnm)
                    )
        ax.set_ylabel(r'$E_{pr}(z)\exp(\alpha z)$ (norm.)')
    ax.set_xlabel('z (km)')
    #ax.legend()
    ax.grid()

def plot_Pbril_vs_z(R):
    fig,ax = plt.subplots(constrained_layout=True)
    #R = [R[i] for i in [0,2,4,6,8,10,12]]
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


def plot_PSDbril_vs_sweepparam(param_vec,R,zi):
    idxz = np.argmin(np.abs(zi-R[0].z))
    fig,ax = plt.subplots(constrained_layout=True)
    y_bril = dbm([r.P_bril[idxz] for r in R])
    y_NEP = dbm([r.NEP for r in R])
    y_b2 = dbm([r.P_b2 for r in R])
    ax.plot(param_vec,y_bril,label=r'$P_{b1}^2$')
    ax.plot(param_vec,y_b2,label=r'$P_{b2}^2$')
    ax.plot(param_vec,y_NEP,label=r'$\sigma_{n}^2$')
    ax.set_xlabel(r'Sweep parameter')
    ax.set_ylabel(r'Power @ '+"{:.1f}".format(R[0].z[idxz]*1e-3)+' km')
    ax.grid()
    ax.legend()

def plot_SNR_vs_z(param_vec,R,Navg=1):
    SNR_max = R[0].SNR_max
    fig,ax = plt.subplots(constrained_layout=True)
    for r in R:
        ax.plot(r.z*1e-3,
                db(r.SNR),#r.SNR*Navg
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

def plot_SNR_vs_sweepparam(param_vec,R,z0=0,dBscale=False,Navg=1,ax=None):
    
    Ppr0_arr = np.array([r.Ppr0 for r in R])
    
    k2 = 1.75*Navg
    SNR_fac = (k2+1)/np.sqrt(2*(2*k2+1))
    
    SNR_arr = np.array([SNR_fac*r.V2_signal/r.V2_noise for r in R])
    
    if ax==None:
        fig,ax = plt.subplots(constrained_layout=True)
    idx_z = np.argmin(np.abs(R[0].z-z0))
    
    print(R[0].z[idx_z])
    
    y = np.array([SNR[idx_z] for SNR in SNR_arr])
    labeltext = 'SNR @ '
    if dBscale:
        y = db([SNR[idx_z] for SNR in SNR_arr])
        labeltext = 'SNR (dB) @ '
    ax.plot(Ppr0_arr*1e3,y,label='model (with MI)',color='tab:green')
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
    subfolder_path = file_dir+r"\data\MI_test\meas_compare_dfm\noise_calculated_fine"

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
    z0 = 250e3
    Navg = 1024
    plt.close('all')
    plot_Pinband_vs_z(R,plot_norm=True)
    plot_Pbril_vs_z(R)
    plot_PSD_vs_lam(R,z0=z0)
    plot_PSDnoise_vs_z(param_vec,R)
    plot_SNR_vs_z(param_vec,R,Navg=Navg)
    plot_PSDbril_vs_sweepparam(param_vec,R,z0)
    plot_SNR_vs_sweepparam(param_vec,R,z0=z0,Navg=Navg,dBscale=True)
    plot_PSD_at_zi(R[-1],0)
    plt.show()

# %%
