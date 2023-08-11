

# %% Import modules
import sys
import os
import time
import multiprocessing
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
from src.simulation_system import Simulation_pulsed_sections2_fiber
from help_functions import dbm,db,norm_fft, moving_average, lorentzian

# %% Define propagation fibers
def A0_func(t,T0,Ppeak0):
    return sqrt(Ppeak0)*exp(-(2*t/T0)**22)

L = 75e3                # Fiber length (km)
T0 = 100                # Pulse length (ns)
Fiber2 = Fiber_SumULL
Fiber1 = Fiber_TWXL
Ppeak0 = 150e-3
PSDnoise_dbmHz = -141   # Noise floor (dBm/Hz)

Tmax = T0*7             # Simulation window size (ns)
N = 2**16
t = np.linspace(-Tmax/2,Tmax/2,N)
Nz_save = 21
Nsec = 3

step_sweep = 5
PSDnoise_dbmHz_vec = -141

L1_vec = np.arange(0,75,10)*1e3
L2_vec = L-L1_vec

N_sweep = len(L1_vec)

# %% Run simulation
savedir = r'C:\Users\madshv\OneDrive - Danmarks Tekniske Universitet\code\system_optimization\data\MI_test\altfiber_sec3'

def sim_func(args):
    i, Ppeak0, t, T0, L1, L2, Nz_save, Fiber1, Fiber2, PSDnoise_dbmHz, Nsec, savedir = args
    print('Start:\t Simulation no. '+str(i))
    print(L1,L2)
    start_time = time.time()
    A0 = A0_func(t, T0, Ppeak0)
    S = Simulation_pulsed_sections2_fiber(t, A0, L1, L2, Nz_save, Fiber1, Fiber2, 
                                          PSDnoise_dbmHz, Nsec)
    z, A = S.run()
    #savefname = f'P0_{int(Ppeak0 * 1000)}.pkl'
    savefname = f'L_{int(L1*1e-3)}.pkl'
    S.save_pickle(savedir, savefname)
    end_time = time.time()
    print('End:\t Simulation no. '+str(i)+' Time: '+str(end_time-start_time))

if __name__ == '__main__':
    args_list = [(i, Ppeak0, t, T0, L1_vec[i],L2_vec[i], Nz_save, Fiber1,Fiber2, 
                  PSDnoise_dbmHz_vec, Nsec, savedir) for i in range(N_sweep)]
    with multiprocessing.Pool() as pool:
        pool.map(sim_func, args_list)
                 

# %%


