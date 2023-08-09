


# %% Import modules
import sys
import os
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
sys.path.insert(0, 'C:/Users/madshv/OneDrive - Danmarks Tekniske Universitet/code')

import numpy as np
import pickle
from numpy import sqrt,exp
import matplotlib.pyplot as plt
from physical_parameters import *
from numpy.fft import fft,fftshift
from scipy.signal import convolve
from src.simulation_system import Simulation_pulsed_sections_fiber
from help_functions import dbm,db,norm_fft, moving_average, lorentzian

# %% Define propagation fibers
def A0_func(t,T0,Ppeak0):
    return sqrt(Ppeak0)*exp(-(2*t/T0)**22)

L = 100e3                # Fiber length (km)
T0 = 100                # Pulse length (ns)
Fiber = Fiber_SumULL
PSDnoise_dbmHz = -141   # Noise floor (dBm/Hz)

Tmax = T0*7             # Simulation window size (ns)
N = 2**16
t = np.linspace(-Tmax/2,Tmax/2,N)
Nz_save = 101
Nsec = 1

step_sweep = 20
Ppeak0_vec = np.arange(50,300,step_sweep)*1e-3      # Pump power (W)
N_sweep = len(Ppeak0_vec)

# %% Run simulation
savedir = r'C:\Users\madshv\OneDrive - Danmarks Tekniske Universitet\code\system_optimization\data\MI_test\sec3'

for i in range(N_sweep):
    Ppeak0 = Ppeak0_vec[i]
    A0 = A0_func(t,T0,Ppeak0)
    S = Simulation_pulsed_sections_fiber(t,A0,L,Nz_save,Fiber,PSDnoise_dbmHz,Nsec)
    z,A = S.run()
    savefname = 'P0_'+str(int(Ppeak0*1000))+'.pkl'
    S.save_pickle(savedir,savefname)


# %%

