# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 11:47:10 2022

@author: madshv
"""
import sys
sys.path.insert(0, 'C:/Users/madshv/OneDrive - Danmarks Tekniske Universitet/code/system_optimization/src')
import numpy as np
from solver_system import System_solver_class
from scipy.constants import c
from numpy import pi
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