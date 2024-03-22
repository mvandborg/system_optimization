
# %% Import modules
import os
import sys
this_dir = os.path.dirname(__file__)
sys.path.append(this_dir)

import time
import multiprocessing

import numpy as np
from numpy import sqrt,exp
from src.fiberdata_passive import Passivefiber_class
from src import simulation_system as simsys
from src.help_functions import PSD_dbmGHz2dbmnm, PSD_dbmnm2dbmGHz

# %% Define propagation fibers
def A0_func(t,T0,Ppeak0):
    return sqrt(Ppeak0)*exp(-(2*t/T0)**22)

# Define physical parameters
L = 75e3                # Fiber length (km)
T0 = 100                # Pulse length (ns)

lam_p = 1455e-9         # Wavelength (m)
lam_pr = 1550e-9
lam_arr = np.array([lam_p,lam_pr])
Ppeak0 = 100e-3
PSD_noise_dbmnm = -30
PSDnoise_dbmGHz = PSD_dbmnm2dbmGHz(PSD_noise_dbmnm,lam_pr*1e9,2.998e8)

fiberdata_path = os.path.join(this_dir, 'fiber_data')
Fiber1 = Passivefiber_class.from_data_sheet(fiberdata_path,
                                            'OFS_SCUBA150.json',
                                            lam_arr)
Fiber2 = Passivefiber_class.from_data_sheet(fiberdata_path,
                                            'OFS_TruewaveXL.json',
                                            lam_arr)


# Define numerical parameters
Tmax = T0*7             # Simulation window size (ns)
N = 2**16
t = np.linspace(-Tmax/2,Tmax/2,N)
Nz_save = 21
Nsec = 3

L1_vec = np.linspace(1e-3,89.99,24)*1e3
L2_vec = L-L1_vec

N_sweep = len(L1_vec)

Fiber_arr = [[Fiber1],[Fiber2,Fiber1],[Fiber1]]
L_arr = [[[90e3],[L1_vec[i],90e3-L1_vec[i]],[100e3]] for i in range(len(L1_vec))]

# %% Run simulation
# Insert directory for saving data
savedir = this_dir+r'\data\MI_test\TWXLinsec2_sec3\P100'

def sim_func(args):
    i, Ppeak0, t, T0, L_arr, Nz_save, Fiber_arr, PSDnoise_dbmGHz, Nsec, savedir = args
    print('Start:\t Simulation no. '+str(i))
    start_time = time.time()
    A0 = A0_func(t, T0, Ppeak0)
    S = simsys.Simulation_pulsed_customsections2(t, A0, L_arr, Nz_save, Fiber_arr, 
                                                  PSDnoise_dbmGHz)
    z, A = S.run()
    #savefname = f'P0_{int(Ppeak0 * 1000)}.pkl'
    savefname = f'L_{int(L_arr[1][0]*1e-3)}.pkl'
    S.save_pickle(savedir, savefname)
    end_time = time.time()
    print('End:\t Simulation no. '+str(i)+' Time: '+str(end_time-start_time))

if __name__ == '__main__':
    args_list = [(i, Ppeak0, t, T0, L_arr[i], Nz_save, Fiber_arr, 
                  PSDnoise_dbmGHz, Nsec, savedir) for i in range(N_sweep)]
    #for args in args_list:
    #    sim_func(args)
    with multiprocessing.Pool() as pool:
        pool.map(sim_func, args_list)
                 

# %%


