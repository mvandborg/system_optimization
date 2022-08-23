# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 11:48:25 2022

@author: madshv
"""

import numpy as np
from numpy import pi
from scipy.constants import c
c = c*1e-9              # Unit m/ns

class Passivefiber_class:
    def __init__(self,lam,alpha_db,D,Aeff):
        self.n2 = 2.6e-20                       # Nonlinear coefficient m2/W  
        
        self.lam = lam                          # Wavelengths (m)
        self.alpha_db = alpha_db                # Loss in dB/km
        self.alpha = alpha_db*1e-3/4.34         # Loss in 1/m
        
        self.D = D                              # GVD param (ps/(nm*km))
        self.D = D*1e3                          # Convert from ps/nm*km to ns/m2
        self.beta2 = -lam**2/(2*pi*c)*self.D    # Unit ns2/m
        
        self.Aeff = Aeff                        # Effective area (1/m2)    
        
        self.f = c/self.lam                     # Frequency (GHz)
        self.omega = 2*pi*self.f                # Angular frequency (GHz)
        self.gamma = self.n2*self.omega/(c*self.Aeff)   # Nonlinear coefficient (1/m)      
        
        
    def add_raman(self,df,gr):
        self.fr = 0.18
        self.df_raman = df
        self.gr = gr                            # Raman coefficient (1/(W*m))
        
        