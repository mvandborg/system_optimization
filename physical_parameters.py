# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 15:39:11 2022

@author: madshv
"""

# %% Imports

# Import path to erbium model
import sys
import os
sys.path.append(os.path.dirname(__file__))  # Insert current folder to path

from numpy import pi

c = 2.998              # Unit m/ns

# Define physical parameters
lam_p = 1455e-9         # Wavelength (m)
lam_pr = 1550e-9

f_p = c/lam_p           # Frequencies (GHz)
f_pr = c/lam_pr

omega_p = 2*pi*f_p
omega_pr = 2*pi*f_pr

f_delta = f_p-f_pr


# Brillouin parameters
T = 300                 # Temperature (K)
f_b = 10.8              # Brillouin frequency shift (GHz)
FWHM_b = 38e-3          # FWHM of Brillouin peak (GHz)
Gamma_b = 2*pi*FWHM_b   # Decay factor (1/ns)
g_b = 0.1470            # Brillouin gain factor(1/W/m)

gamma_raman = 1.72e-14  # Raman gain coefficient (m/W)
df_raman = f_p-f_pr


