# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 16:26:55 2021

@author: madshv
"""

from numpy import log10
import numpy as np
from numpy.fft import fft,ifft,fftshift

NP_TO_DB = 4.34

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
    return fftshift(ifft(AF)/dt)

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