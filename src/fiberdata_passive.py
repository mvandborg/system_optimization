# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 11:48:25 2022

@author: madshv
"""

import numpy as np
from numpy import pi
import json
import os

c_m_ns = 0.2998             # Unit m/ns
c_km_ps = 0.2998e-6         # Unit km/ps
NP_TO_DB = 4.34

class Passivefiber_class:
    def __init__(self,lam,alpha_db=0.2,D=17,Aeff=80,ng=1.46,name='NA'):
        self.name = name        
        self.n2 = 2.6e-20                       # Nonlinear coefficient m2/W  
        
        self.lam = lam                          # Wavelengths (m)
        self.alpha_db = alpha_db                # Loss in dB/km
        self.alpha = alpha_db*1e-3/NP_TO_DB     # Loss in 1/m
        
        self.ng = ng
        self.vg = c_m_ns/ng                     # Group velocity (m/ns)
        
        self.D = D                              # GVD param (ps/(nm*km))
        self.D = D*1e3                          # Convert from ps/nm*km to ns/m2
        self.beta2 = -lam**2/(2*pi*c_m_ns)*self.D    # Unit ns2/m
        
        self.Aeff = Aeff*1e-12                  # Effective area (1/m2)    
        
        self.f = c_m_ns/self.lam                # Frequency (GHz)
        self.omega = 2*pi*self.f                # Angular frequency (GHz)
        self.gamma = self.n2*self.omega/(c_m_ns*self.Aeff)   # Nonlinear coefficient (1/W*m)      
        
    def add_raman(self,df,gr):
        self.fr = 0.18
        self.df_raman = df
        self.gr = gr                            # Raman coefficient (1/(W*m))

    @classmethod
    def from_data_sheet(cls,file_path,filename,lam):
        # lam = wavelength in m
        
        lam_nm = lam*1e9            # Convert to nm
        
        # Load the data from the .json file
        full_path = os.path.join(file_path, filename)
        with open(full_path, 'r') as json_file:
            dat = json.load(json_file)
        lam_dat = dat['wav']
        alpha_db_dat = dat['alpha_db']
        D_dat = dat['GVD']
        S_dat = dat['Disp_slope']
        Aeff_dat = dat['Aeff']
        ng_dat = dat['ng']
        name = dat['name']
        
        if isinstance(lam,(list,np.ndarray)):
            N = len(lam)
        else: 
            N = 1
        
        # Interpolate the data to the given wavelengths (in m)
        dlam_nm = lam_nm-lam_dat
        
        D = D_dat+dlam_nm*S_dat    # Calculate D for the specified wavelengths
        Aeff = Aeff_dat*np.ones(N)  # Assume constant Aeff
        ng = ng_dat+c_km_ps*(D_dat*dlam_nm+S_dat*dlam_nm**2)
        
        A2 = 8.87e11     # alpha = A1+A2/lam**4 (unit dB/nm**4)
        alpha_db = alpha_db_dat+A2*(lam_nm**-4-lam_dat**-4)
                
        return cls(lam,alpha_db=alpha_db,D=D,Aeff=Aeff,ng=ng,name=name)

