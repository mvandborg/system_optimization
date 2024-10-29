# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 11:47:10 2022

@author: madshv
"""
import os
import numpy as np
import pickle
from .solver_system import System_solver_class,gnls1
from scipy.constants import c
from scipy.constants import h as h_planck
from scipy.signal import butter,freqz
from numpy import pi
from .help_functions import dbm, inv_dbm, norm_fft, norm_ifft,\
    PSD2ESD,PSD_dbmGHz2dbmnm
from numpy.fft import fft,ifft,fftfreq,fftshift
c = c*1e-9              # Unit m/ns

def get_PSD_quant_noise_WGHz():
    lam_0 = 1550e-9
    c_m_s = 2.998e8
    nu_0 = c_m_s/lam_0  # Frequency (Hz)
    PSD_QL_WGHz = 1/2*h_planck*nu_0*1e9   # ESD of quantum noise floor (W/GHz)
    return PSD_QL_WGHz

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
        
    def add_section(self,L_co,L_edf,L_fib,Fiber_co,
                    Fiber_edf,Fiber_fib,Fiber_pd,C):
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
        self.lam_noise_init = np.linspace(lam_noise_min,
                                          lam_noise_max,Nlamnoise+1)
        self.lam_noise = self.lam_noise_init[0:-1]
        self.f_noise = c/self.lam_noise
        self.BW_noise = c*(1/self.lam_noise_init[0:-1]\
            -1/self.lam_noise_init[1:])
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
    def __init__(self,t,A0,L,Nz_save,Fiber,PSDnoise_dbmGHz=-71.9,linewidth=0):
        # Pnoise=-72.9 dBm/GHz is the quantum limit (ESD=1/2*h*nu)
        self.t = t
        self.A0 = A0
        self.L = L
        self.N = len(t)
        self.Tmax = t[-1]-t[0]
        self.A0 = A0
        self.Nz_save = Nz_save
        self.Fiber = Fiber
        self.PSDnoise_dbmGHz = PSDnoise_dbmGHz
        self.PSDnoise_dbmHz = PSDnoise_dbmGHz-90
        self.PSDnoise_dbmnm = PSD_dbmGHz2dbmnm(PSDnoise_dbmGHz,1550,3e8)
        self.dnu = linewidth    # Phase noise linewidth (GHz)
        
        self.dt = self.Tmax/self.N
        self.fsam = 1/self.dt         # Sampling frequency (GHz)
        self.fnyq = self.fsam/2
        
        self.f = fftfreq(self.N,d=self.dt)
        self.df = self.f[1]-self.f[0]
        self.omega = 2*pi*self.f
        self.omega_sh = fftshift(self.omega)
        self.f_sh = self.omega_sh/(2*pi)
               
        # Energy spectral density of the noise (nJ/GHz)
        self.ESDnoise_nJGHz = PSD2ESD(inv_dbm(self.PSDnoise_dbmGHz),self.Tmax)
        
        # Amplitude spectral density (sqrt(nJ/GHz))
        # Random phase and amplitude
        Xre = np.random.normal(size=self.N)
        Xim = np.random.normal(size=self.N)
        self.ASDnoise = np.sqrt(self.ESDnoise_nJGHz/2)*(Xre+1j*Xim)
        #self.ASDnoise = np.sqrt(self.ESDnoise_nJGHz)*\
        #    np.exp(2j*np.pi*np.random.random(self.N))
        
        # Phase noise
        phase = np.zeros(self.N)
        fac = np.sqrt(2*np.pi*self.dnu*self.dt)
        for i in range(1,self.N):
            phase[i] = phase[i-1] + fac*np.random.normal()     
                    
        # Normalized such that |AF|^2=ESD and sum(|AF|^2)*df=sum(|A|^2)*dt
        self.Anoise = norm_ifft(self.ASDnoise,self.dt)              

        self.A0 = self.A0*np.exp(1j*phase)+self.Anoise      # Add noises
        self.AF0 = norm_fft(self.A0,self.dt)

        # Filter the input signal
        add_filter = False
        if add_filter:
            fcut_filt = 28  # Filter cut off frequency (GHz)
            order_filt = 5

            b,a = butter(order_filt,fcut_filt,fs=self.fsam)
            w,h = freqz(b, a, fs=self.fsam, worN=self.N,whole=True)
            
            # Quantum noise
            Xre_QL = np.random.normal(size=self.N)
            Xim_QL = np.random.normal(size=self.N)
            # ESD of quantum noise floor (W/GHz)
            PSD_QL_WGHz = get_PSD_quant_noise_WGHz()   
            ESD_QL_nJGHz = PSD2ESD(PSD_QL_WGHz,self.Tmax)
            ASD_QL = np.sqrt(ESD_QL_nJGHz/2)*(Xre_QL+1j*Xim_QL)
            
            self.AF0 = fftshift(h)*self.AF0+ASD_QL
            self.A0 = norm_ifft(self.AF0,self.dt)
        
    
    def run(self):
        z,A = gnls1(self.t,self.omega,self.A0,self.L,self.Fiber,self.Nz_save)
        self.z = z
        self.A = A
        return z,A

    def save_pickle(self,filedir,filename):
        if type(self.Fiber)==list:
            if type(self.Fiber[0])==list:
                fiber_list = [j for sub in self.Fiber for j in sub]
            else:
                fiber_list = self.Fiber
            savedict = {
                "Fiber_dict":[x.__dict__ for x in fiber_list],
                "A":self.A,
                "z":self.z,
                "f":self.f_sh,
                "t":self.t,
                "L":self.L,
                "PSDnoise_dbmHz":self.PSDnoise_dbmHz,
                "PSDnoise_dbmGHz":self.PSDnoise_dbmGHz,
                "linewidth":self.dnu
                }
        else:
            savedict = {
                "Fiber_dict":self.Fiber.__dict__,
                "A":self.A,
                "z":self.z,
                "f":self.f_sh,
                "t":self.t,
                "L":self.L,
                "PSDnoise_dbmHz":self.PSDnoise_dbmHz,
                "PSDnoise_dbmGHz":self.PSDnoise_dbmGHz,
                "linewidth":self.dnu
                }
        with open(os.path.join(filedir,filename), "wb") as f:
            pickle.dump(savedict, f)
        
# Class of multiple section fiber propagation for MI investigation
class Simulation_pulsed_sections_fiber(Simulation_pulsed_single_fiber):
    def __init__(self,t,A0,L,Nz_save,Fiber,Nsec,
                 PSDnoise_dbmGHz=-72.9,linewidth=0):
        super().__init__(t,A0,L,Nz_save,Fiber,PSDnoise_dbmGHz,linewidth)
        self.Nsec = Nsec
    
    def run(self):
        Atot = []
        ztot = []
        z1,A1 = gnls1(self.t,self.omega,self.A0,
                      self.L,self.Fiber,self.Nz_save)
        Atot = A1
        ztot = z1
        if type(self.Fiber.alpha)==list:
            G = np.exp(self.Fiber.alpha[1]*self.L)
        else:
            G = np.exp(self.Fiber.alpha*self.L)
        for i in range(1,self.Nsec):
            A0_new = np.sqrt(G)*A1[:,-1]
            z1,A1 = gnls1(self.t,self.omega,A0_new,self.L,
                          self.Fiber,self.Nz_save)
            Atot = np.concatenate([Atot,A1[:,1:]],axis=1)
            ztot = np.concatenate([ztot,ztot[-1]+z1[1:]])
        self.z = ztot
        self.A = Atot
        return ztot,Atot
    

# Class of multiple section fiber propagation for MI investigation
class Simulation_pulsed_sections2_fiber(Simulation_pulsed_single_fiber):
    def __init__(self,t,A0,L1,L2,Nz_save,Fiber1,Fiber2,PSDnoise_dbmHz,Nsec,
                 linewidth=0):
        super().__init__(t,A0,[L1,L2],Nz_save,[Fiber1,Fiber2],PSDnoise_dbmHz,
                         linewidth=linewidth)
        self.Nsec = Nsec
    
    def run(self):
        Lsec = np.sum(self.L)
        G = np.exp(self.Fiber[0].alpha*self.L[0]\
            +self.Fiber[1].alpha*self.L[1])
        
        Atot = []
        ztot = []
        
        # First propagation
        z1,A1 = gnls1(self.t,self.omega,self.A0,self.L[0],
                      self.Fiber[0],self.Nz_save)
        Atot = A1
        ztot = z1
        A0_new = A1[:,-1]
        z1,A1 = gnls1(self.t,self.omega,A0_new,self.L[1],
                      self.Fiber[1],self.Nz_save)
        Atot = np.concatenate([Atot,A1[:,1:]],axis=1)
        ztot = np.concatenate([ztot,ztot[-1]+z1[1:]])
        
        # Next propagations
        for i in range(1,self.Nsec):
            A0_new = np.sqrt(G)*A1[:,-1]
            z1,A1 = gnls1(self.t,self.omega,A0_new,self.L[0],
                          self.Fiber[0],self.Nz_save)
            Atot = np.concatenate([Atot,A1[:,1:]],axis=1)
            ztot = np.concatenate([ztot,ztot[-1]+z1[1:]])
            A0_new = A1[:,-1]
            z1,A1 = gnls1(self.t,self.omega,A0_new,self.L[1],
                          self.Fiber[1],self.Nz_save)
            Atot = np.concatenate([Atot,A1[:,1:]],axis=1)
            ztot = np.concatenate([ztot,ztot[-1]+z1[1:]])
        self.z = ztot
        self.A = Atot
        return ztot,Atot

# Class of multiple section fiber propagation for MI invesigation.
# Here, the sections are not repeated, but each section is customizable

class Simulation_pulsed_customsections2(Simulation_pulsed_single_fiber):
    # L is an N x 2 array, with N sections of two lengths.
    # Fiber is an N x 2 array of Fiber classes.
    def __init__(self,t,A0,L_arr,Nz_save,Fiber_arr,PSDnoise_dbmHz,linewidth=0):
        super().__init__(t,A0,L_arr,Nz_save,Fiber_arr,PSDnoise_dbmHz,
                         linewidth=linewidth)
        self.Nsec = len(L_arr)

    def run(self):
        # First propagation         
        z1,A1 = gnls1(self.t,self.omega,self.A0,self.L[0][0],
                      self.Fiber[0][0],self.Nz_save)
        Atot = A1
        ztot = z1
        A0_new = self.A0
        for j in range(1,len(self.L[0])):
            z1,A1 = gnls1(self.t,self.omega,A0_new,self.L[0][j],
                          self.Fiber[0,j],self.Nz_save)
            A0_new = A1[:,-1]
            Atot = np.concatenate([Atot,A1[:,1:]],axis=1)
            ztot = np.concatenate([ztot,ztot[-1]+z1[1:]])
        
        # Next propagations
        for i in range(1,self.Nsec):
            # Calculate the loss of the former section
            loss_neper = np.sum([self.Fiber[i-1][j].alpha[1]*self.L[i-1][j]\
                for j in range(len(self.L[i-1]))])
            G = np.exp(loss_neper)
            A0_new = np.sqrt(G)*A1[:,-1]
            for j in range(0,len(self.L[i])):
                z1,A1 = gnls1(self.t,self.omega,A0_new,self.L[i][j],
                            self.Fiber[i][j],self.Nz_save)
                A0_new = A1[:,-1]
                Atot = np.concatenate([Atot,A1[:,1:]],axis=1)
                ztot = np.concatenate([ztot,ztot[-1]+z1[1:]])
        self.z = ztot
        self.A = Atot
        return ztot,Atot
    
    