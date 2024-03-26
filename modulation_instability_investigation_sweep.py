
# %% Import modules
import os
import sys
this_dir = os.path.dirname(__file__)
sys.path.append(this_dir)

import time
import multiprocessing
import numpy as np
from numpy import sqrt,exp
from physical_parameters import *
from src.simulation_system import Simulation_pulsed_sections_fiber
from src.help_functions import PSD_dbmGHz2dbmnm, PSD_dbmnm2dbmGHz

# %% Define propagation fibers
def A0_func(t,T0,Ppeak0):
    return sqrt(Ppeak0)*exp(-(2*t/T0)**22)

# Directory for saving the data

savedir = this_dir+r'\data\MI_test\altfiber_sec1\P200'

L = 75e3                # Fiber length (km)
T0 = 100                # Pulse length (ns)
Fiber = Fiber_Scuba150
#Ppeak0 = 150e-3
PSD_noise_dbmnm = -30
PSDnoise_dbmGHz = PSD_dbmnm2dbmGHz(PSD_noise_dbmnm,1550,2.99e8)

Tmax = T0*7             # Simulation window size (ns)
N = 2**16
t = np.linspace(-Tmax/2,Tmax/2,N)
Nz_save = 21
Nsec = 3

step_sweep = 5
Ppeak0_vec = np.arange(10,120,step_sweep)*1e-3      # Pump power (W)
#PSDnoise_dbmGHz_vec = np.array([-151,-141,-131,-121])+90
N_sweep = len(Ppeak0_vec)

# %% Run simulation
def sim_func(args):
    i, Ppeak0, t, T0, L, Nz_save, Fiber, PSDnoise_dbmGHz, Nsec, savedir = args
    print('Start:\t Simulation no. '+str(i))
    start_time = time.time()
    A0 = A0_func(t, T0, Ppeak0)
    S = Simulation_pulsed_sections_fiber(t, A0, L, Nz_save, Fiber, PSDnoise_dbmGHz, Nsec)
    z, A = S.run()
    #savefname = f'P0_{int(Ppeak0 * 1000)}.pkl'
    savefname = f'P0_{int(-PSDnoise_dbmGHz)}.pkl'
    #S.save_pickle(savedir, savefname)
    end_time = time.time()
    print('End:\t Simulation no. '+str(i)+' Time: '+str(end_time-start_time))

if __name__ == '__main__':
    args_list = [(i, Ppeak0_vec[i], t, T0, L, Nz_save, Fiber, 
                  PSDnoise_dbmGHz, Nsec, savedir) for i in range(N_sweep)]
    with multiprocessing.Pool() as pool:
        pool.map(sim_func, args_list)
                 

# %%

