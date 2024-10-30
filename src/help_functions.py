# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 16:26:55 2021

@author: madshv
"""

import json
import os

from numpy import log10
import numpy as np
from numpy.fft import fft,ifft,fftshift

NP_TO_DB = 4.34

def load_config():
    config_file_path = os.path.join(os.path.dirname(__file__), 'config.json')

    with open(config_file_path, 'r') as f:
        config = json.load(f)

    return config

def np_to_db(P_np):
    return P_np*NP_TO_DB

def db_to_np(P_db):
    return P_db/NP_TO_DB

def db(x):
    return 10*log10(x)

def inv_db(x):
    return 10**(x/10)

def dbm(x):
    return 10*log10(x)+30

def inv_dbm(x):
    return 10**(x/10)*1e-3

def norm_fft(A,dt):         # Normalized fft such that sum(|A|^2)*dt=sum(|AF|^2)*df
    return fftshift(fft(A)*dt)

def norm_ifft(AF,dt):       # Normalized ifft such that sum(|A|^2)*dt=sum(|AF|^2)*df
    return ifft(fftshift(AF))/dt

def norm_fft2d(A,dt,axis=0):         # Normalized fft such that sum(|A|^2)*dt=sum(|AF|^2)*df
    return fftshift(fft(A,axis=0)*dt,axes=0)

def ESD2PSD(ESD,Tmax):
    # Tmax (unit ns)
    # ESD (unit nJ/GHz)
    return ESD/Tmax         # PSD of rayleigh scattering (W/GHz)

def PSD2ESD(ESD,Tmax):
    # Tmax (unit ns)
    # PSD (W/GHz)
    return ESD*Tmax         # ESD (unit nJ/GHz)
      
def PSD_dbmnm2dbmGHz(PSD_dbm_nm,lam_nm,clight):
    # lam_nm (unit nm)
    # PSD_dbm_nm (unit dBm/nm)
    # clight (unit nm/ns=nm*GHz=m/s)
    PSD_dbm_GHz = PSD_dbm_nm+db(lam_nm**2/clight)
    return PSD_dbm_GHz

def PSD_dbmGHz2dbmnm(PSD_dbm_GHz,lam_nm,clight):
    PSD_dbm_nm = PSD_dbm_GHz-db(lam_nm**2/clight)
    return PSD_dbm_nm

def moving_average(a, n=3):
    ret = np.cumsum(a,axis=0, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def lorentzian(f,f0,fwhm):
    return (fwhm/2)**2/((f-f0)**2+(fwhm/2)**2)

def get_ASE_noise_WGHz(G=20,lam0=1550e-9):
    n_sp = 1.5
    h = 6.626e-34
    c = 2.998e8
    nu = c/lam0
    PSD_ase = n_sp*h*nu*(G-1) # PSD of ASE (W/Hz)
    return PSD_ase*1e9

def get_ASE_ql(lam0=1550e-9):
    h = 6.626e-34
    c = 2.998e8
    nu = c/lam0
    PSD_ase = h*nu # PSD of ASE (W/Hz)
    return PSD_ase

def get_ASE_noise_dbmnm(G,lam_pr):
    PSD_ase = get_ASE_noise_WGHz(G=1000,lam0=lam_pr)
    PSD_ase_dbmGHz = dbm(PSD_ase)
    PSDnoise_dbmGHz = PSD_ase_dbmGHz
    PSD_noise_dbmnm = PSD_dbmGHz2dbmnm(PSDnoise_dbmGHz,lam_pr*1e9,2.998e8)
    return PSD_noise_dbmnm