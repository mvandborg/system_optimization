# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 11:47:10 2022

@author: madshv
"""
import sys
sys.path.insert(0, 'C:/Users/madshv/OneDrive - Danmarks Tekniske Universitet/code/system_optimization/src')
sys.path.insert(0, 'C:/Users/madshv/OneDrive - Danmarks Tekniske Universitet/code')
import numpy as np
import pickle
from solver_system import System_solver_class,gnls1
from scipy.constants import c
from numpy import pi
from help_functions import inv_dbm, inv_norm_fft, norm_fft
from numpy.fft import fft,ifft,fftfreq,fftshift
c = c*1e-9              # Unit m/ns

# Class of CW simulation
class System_simulation_class:
    def __init__(self,lam_p,lam_pr,Ppr0,Pp0,L0,Fiber_fib0,Fiber_pd0,Nz,\
                 Tpulse,Temp,f_b,FWHM_b,ng,g_b):
        self.lam_p = lam_p
        self.lam_pr = lam_pr
        self.f_p = c/self.lam_p
        self.f_pr = c/self.lam_pr
        self.omega_p = 2*pi*self.f_p
        self.omega_pr = 2*pi*self.f_pr
        self.Ppr0 = Ppr0
        self.Pp0 = Pp0
        
        self.L0 = L0
        self.Fiber_fib0 = Fiber_fib0
        self.Fiber_pd0 = Fiber_pd0
        
        self.L_co = np.array([])
        self.L_edf = np.array([])
        self.L_fib = np.array([])
        
        self.Fiber_co = np.array([])
        self.Fiber_edf = np.array([])
        self.Fiber_fib = np.array([])
        self.Fiber_pd = np.array([])
        
        self.C = np.array([])           # Coupling factor 0<C<1
        self.Nsec = 0
        
        self.lam_noise_init = np.array([])
        self.lam_noise = np.array([])
        self.f_noise = np.array([])
        self.BW_noise = np.array([])
        
        self.Nz = Nz
        
        # Brillouin scattering parameters
        self.Tpulse = Tpulse
        self.Temp = Temp
        self.f_b = f_b
        self.FWHM_b = FWHM_b
        self.Gamma_b = 2*pi*self.FWHM_b
        self.ng = ng
        self.g_b = g_b
        self.vg = c/ng
        
    def add_section(self,L_co,L_edf,L_fib,Fiber_co,Fiber_edf,Fiber_fib,Fiber_pd,C):
        self.Nsec+=1
        self.L_co = np.append(self.L_co,L_co)
        self.L_edf = np.append(self.L_edf,L_edf)
        self.L_fib = np.append(self.L_fib,L_fib)
        
        self.Fiber_co = np.append(self.Fiber_co,Fiber_co)
        self.Fiber_edf = np.append(self.Fiber_edf,Fiber_edf)
        self.Fiber_fib = np.append(self.Fiber_fib,Fiber_fib)
        self.Fiber_pd = np.append(self.Fiber_pd,Fiber_pd)
        
        self.C = np.append(self.C,C)

    
    def add_noise(self,lam_noise_min,lam_noise_max,Nlamnoise):
        self.lam_noise_init = np.linspace(lam_noise_min,lam_noise_max,Nlamnoise+1)
        self.lam_noise = self.lam_noise_init[0:-1]
        self.f_noise = c/self.lam_noise
        self.BW_noise = c*(1/self.lam_noise_init[0:-1]-1/self.lam_noise_init[1:])
        self.idx_noise = np.argmin(np.abs(self.lam_noise-self.lam_pr))
    
    def run(self):
        Sol = System_solver_class(self)
        return Sol.run()


# Class of Pulsed system simulation
class System_simulation_pulsed_class(System_simulation_class):
    def __init__(self,t,Apr0,Ap0,lam_p,lam_pr,L0,Fiber_fib0,Fiber_pd0,Nz,
                 Tpulse,Temp,f_b,FWHM_b,ng,g_b):
        self.Apr0 = Apr0
        self.Ap0 = Ap0
        self.t = t
        Ppr0 = np.max(Apr0)**2
        Pp0 = np.max(Ap0)**2
        super().__init(self,lam_p,lam_pr,Ppr0,Pp0,L0,Fiber_fib0,Fiber_pd0,Nz,
                       Tpulse,Temp,f_b,FWHM_b,ng,g_b)
        
        
# Class of single fiber propagation for MI investigation
class Simulation_pulsed_single_fiber:
    def __init__(self,t,A0,L,Nz_save,Fiber,PSDnoise_dbmHz):
        self.t = t
        self.A0 = A0
        self.L = L
        self.N = len(t)
        self.Tmax = t[-1]-t[0]
        self.A0 = A0
        self.Nz_save = Nz_save
        self.Fiber = Fiber
        self.PSDnoise_dbmHz = PSDnoise_dbmHz
                
        self.dt = self.Tmax/self.N
        self.fsam = 1/self.dt         # Sampling frequency (GHz)
        self.fnyq = self.fsam/2
        
        self.f = fftfreq(self.N,d=self.dt)
        self.df = self.f[1]-self.f[0]
        self.omega = 2*pi*self.f
        self.omega_sh = fftshift(self.omega)
        self.f_sh = self.omega_sh/(2*pi)
        
        self.C_capture = 1.5e-3
        self.C_loss = 4.5e-3
        self.T0 = 100       # NEEDS TO BE FIXED
        self.Lpulse = c/1.45*self.T0
        # Rayleigh scattering coefficient (unitless)
        self.C_rayscat = self.C_capture*self.C_loss*(self.Lpulse/2)    
        # Brillouin scattering coefficient (unitless)
        self.C_bril = 8e-9   
        
        PSDnoise_WHz = inv_dbm(self.PSDnoise_dbmHz)              # Noise floor (W/Hz)
        PSDnoise_WGHz = PSDnoise_WHz*1e9                        # Noise floor (W/GHz)
        self.ESDnoise_JGHz = PSDnoise_WGHz*self.Tmax                  # Energy spectral density of the noise (nJ/GHz)
        
        self.theta_noise = pi*np.random.uniform(size=self.N)
        self.ASDnoise = np.sqrt(self.ESDnoise_JGHz)*np.exp(1j*self.theta_noise)    # Amplitude spectral density (sqrt(nJ/Hz))
        self.Anoise = ifft(inv_norm_fft(self.ASDnoise,self.dt))                 # Normalized such that |AF|^2=ESD and sum(|AF|^2)*df=sum(|A|^2)*dt

        self.A0 = self.A0+self.Anoise
        self.AF0 = norm_fft(fftshift(fft(self.A0)),self.dt)
    
    def run(self):
        z,A = gnls1(self.t,self.omega,self.A0,self.L,self.Fiber,self.Nz_save)
        self.z = z
        self.A = A
        return z,A

    def save_pickle(self,filedir,filename):
        savedict = {
            "A":self.A,
            "z":self.z,
            "f":self.f_sh,
            "t":self.t,
            "Fiber":self.Fiber,
            "L":self.L,
            "T0":self.T0,
            "PSDnoise_dbmHz":self.PSDnoise_dbmHz,
        }
        with open(filedir+'/'+filename, "wb") as f:
            pickle.dump(savedict, f)
        
# Class of multiple section fiber propagation for MI investigation
class Simulation_pulsed_sections_fiber(Simulation_pulsed_single_fiber):
    def __init__(self,t,A0,L,Nz_save,Fiber,PSDnoise_dbmHz,Nsec):
        super().__init__(t,A0,L,Nz_save,Fiber,PSDnoise_dbmHz)
        self.Nsec = Nsec
    
    def run(self):
        Atot = []
        ztot = []
        z1,A1 = gnls1(self.t,self.omega,self.A0,self.L,self.Fiber,self.Nz_save)
        Atot = A1
        ztot = z1
        G = np.exp(self.Fiber.alpha[1]*self.L)
        for i in range(1,self.Nsec):
            A0_new = np.sqrt(G)*A1[:,-1]
            z1,A1 = gnls1(self.t,self.omega,A0_new,self.L,self.Fiber,self.Nz_save)
            Atot = np.concatenate([Atot,A1[:,1:]],axis=1)
            ztot = np.concatenate([ztot,ztot[-1]+z1[1:]])
        self.z = ztot
        self.A = Atot
        return ztot,Atot
    

# Class of multiple section fiber propagation for MI investigation
class Simulation_pulsed_sections2_fiber(Simulation_pulsed_single_fiber):
    def __init__(self,t,A0,L1,L2,Nz_save,Fiber1,Fiber2,PSDnoise_dbmHz,Nsec):
        super().__init__(t,A0,[L1,L2],Nz_save,[Fiber1,Fiber2],PSDnoise_dbmHz)
        self.Nsec = Nsec
    
    def run(self):
        Lsec = np.sum(self.L)
        G = np.exp(self.Fiber[0].alpha[1]*self.L[0]+self.Fiber[1].alpha[1]*self.L[1])
        
        Atot = []
        ztot = []
        
        # First propagation
        z1,A1 = gnls1(self.t,self.omega,self.A0,self.L[0],self.Fiber[0],self.Nz_save)
        Atot = A1
        ztot = z1
        A0_new = A1[:,-1]
        z1,A1 = gnls1(self.t,self.omega,A0_new,self.L[1],self.Fiber[1],self.Nz_save)
        Atot = np.concatenate([Atot,A1[:,1:]],axis=1)
        ztot = np.concatenate([ztot,ztot[-1]+z1[1:]])
        
        # Next propagations
        for i in range(1,self.Nsec):
            A0_new = np.sqrt(G)*A1[:,-1]
            z1,A1 = gnls1(self.t,self.omega,A0_new,self.L[0],self.Fiber[0],self.Nz_save)
            Atot = np.concatenate([Atot,A1[:,1:]],axis=1)
            ztot = np.concatenate([ztot,ztot[-1]+z1[1:]])
            A0_new = A1[:,-1]
            z1,A1 = gnls1(self.t,self.omega,A0_new,self.L[1],self.Fiber[1],self.Nz_save)
            Atot = np.concatenate([Atot,A1[:,1:]],axis=1)
            ztot = np.concatenate([ztot,ztot[-1]+z1[1:]])
        self.z = ztot
        self.A = Atot
        return ztot,Atot
