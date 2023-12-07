# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 15:39:11 2022

@author: madshv
"""

# %% Imports

# Import path to erbium model
from scipy.constants import c
from numpy import pi
import numpy as np
from src.fiberdata_passive import Passivefiber_class
c = c*1e-9              # Unit m/ns

# Define physical parameters
lam_p = 1455e-9         # Wavelength (m)
lam_pr = 1550e-9

f_p = c/lam_p           # Frequencies (GHz)
f_pr = c/lam_pr

omega_p = 2*pi*f_p
omega_pr = 2*pi*f_pr

f_delta = f_p-f_pr

T0pr = 10e-3               # Pulse duration (ns)

# Brillouin parameters
T = 300                 # Temperature (K)
f_b = 10.8              # Brillouin frequency shift (GHz)
FWHM_b = 38e-3          # FWHM of Brillouin peak (GHz)
Gamma_b = 2*pi*FWHM_b   # Decay factor (1/ns)
ng = 1.45
vg = c/ng               # Group velocity (m/ns)
g_b = 0.1470            # Brillouin gain factor(1/W/m)

gamma_raman = 1.72e-14  # Raman gain coefficient (m/W)
df_raman = f_p-f_pr

# Define fiber parameters

# Generic fiber
alpha_db_p1 = 0.23         # Fiber loss (dB/km)
alpha_db_pr1 = 0.20
D_p1 = 13                   # GVD param (ps/(nm*km))
D_pr1 = 17.27
Aeff1 = 85e-12
Fiber_generic = Passivefiber_class(
                            np.array([lam_p,lam_pr]),
                            np.array([alpha_db_p1,alpha_db_pr1]),
                            np.array([D_p1,D_pr1]),
                            np.array([Aeff1,Aeff1]),
                            name = 'Generic'
                            )
Fiber_generic.add_raman(df_raman,gamma_raman/Aeff1)

# Sumitomo Z Fiber LL
alpha_db_p1 = 0.195         # Fiber loss (dB/km)
alpha_db_pr1 = 0.161
D_p1 = 13                   # GVD param (ps/(nm*km))
D_pr1 = 17
Aeff1 = 85e-12
Fiber_SumLL = Passivefiber_class(np.array([lam_p,lam_pr]),\
                           np.array([alpha_db_p1,alpha_db_pr1]),\
                           np.array([D_p1,D_pr1]),\
                           np.array([Aeff1,Aeff1]),
                           name = 'Sumitomo Z LL')
Fiber_SumLL.add_raman(df_raman,gamma_raman/Aeff1)
    
# Sumitomo Z-PLUS Fiber ULL
alpha_db_p1 = 0.180         # Fiber loss (dB/km)
alpha_db_pr1 = 0.153
D_p1 = 16                   # GVD param (ps/(nm*km))
D_pr1 = 20
Aeff1 = 112e-12
Fiber_SumULL = Passivefiber_class(np.array([lam_p,lam_pr]),\
                           np.array([alpha_db_p1,alpha_db_pr1]),\
                           np.array([D_p1,D_pr1]),\
                           np.array([Aeff1,Aeff1]),
                           name = 'Sumitomo Z-PLUS ULL')
Fiber_SumULL.add_raman(df_raman,gamma_raman/Aeff1)
    
# Sumitomo Z-PLUS Fiber 150
alpha_db_p1 = 0.180         # Fiber loss (dB/km)
alpha_db_pr1 = 0.150
D_p1 = 17                   # GVD param (ps/(nm*km))
D_pr1 = 21
Aeff1 = 150e-12
Fiber_Sum150 = Passivefiber_class(np.array([lam_p,lam_pr]),\
                           np.array([alpha_db_p1,alpha_db_pr1]),\
                           np.array([D_p1,D_pr1]),\
                           np.array([Aeff1,Aeff1]),
                           name = 'Sumitomo Z-PLUS 150')
Fiber_Sum150.add_raman(df_raman,gamma_raman/Aeff1)

# OFS Scuba 150
alpha_db_p1 = 0.180         # Fiber loss (dB/km)
alpha_db_pr1 = 0.150
D_p1 = 17                   # GVD param (ps/(nm*km))
D_pr1 = 21
Aeff1 = 153e-12
Fiber_Scuba150 = Passivefiber_class(np.array([lam_p,lam_pr]),\
                           np.array([alpha_db_p1,alpha_db_pr1]),\
                           np.array([D_p1,D_pr1]),\
                           np.array([Aeff1,Aeff1]),
                           name = 'OFS Scuba150')
Fiber_Scuba150.add_raman(df_raman,gamma_raman/Aeff1)

# OFS Truewave XL
alpha_db_p1 = 0.23         # Fiber loss (dB/km)
alpha_db_pr1 = 0.20
D_p1 = -11                   # GVD param (ps/(nm*km))
D_pr1 = -3
Aeff1 = 72e-12
Fiber_TWXL = Passivefiber_class(np.array([lam_p,lam_pr]),\
                           np.array([alpha_db_p1,alpha_db_pr1]),\
                           np.array([D_p1,D_pr1]),\
                           np.array([Aeff1,Aeff1]),
                           name = 'Sumitomo Truewave XL')
Fiber_TWXL.add_raman(df_raman,gamma_raman/Aeff1)
"""
dir_edf = r'C:/Users/madshv/OneDrive - Danmarks Tekniske Universitet/fiber_data/ofs_edf/'
file_edf = r'LP980_22841_labversion.s'
#'LP980_22841_labversion','LP980_11841'
Fiber_edf = Erbiumfiber_class.from_ofs_files(dir_edf, file_edf)
"""