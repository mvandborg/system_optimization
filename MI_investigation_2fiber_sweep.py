
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
from src.simulation_system import Simulation_pulsed_sections2_fiber
from src.help_functions import PSD_dbmGHz2dbmnm, PSD_dbmnm2dbmGHz,\
    get_ASE_noise_WGHz,dbm

# %% Define propagation fibers
def A0_func(t,T0,Ppeak0):
    return sqrt(Ppeak0)*exp(-(2*t/T0)**22)

# Define physical parameters
L = 75e3               # Fiber length (m)
T0 = 5                # Pulse length (ns)
lam_pr = 1550e-9
Ppeak0 = 300e-3
dnu = 1e-9              # Linewidth of the laser (GHz)
PSD_ase = get_ASE_noise_WGHz(G=1000,lam0=lam_pr)
PSD_ase_dbmGHz = dbm(PSD_ase)
PSDnoise_dbmGHz = PSD_ase_dbmGHz
PSD_noise_dbmnm = PSD_dbmGHz2dbmnm(PSDnoise_dbmGHz,lam_pr*1e9,2.998e8)
PSDnoise_dbmGHz = -51
fiberdata_path = os.path.join(this_dir, 'fiber_data')
Fiber1 = Passivefiber_class.from_data_sheet(fiberdata_path,
                                            'OFS_SCUBA150.json',
                                            lam_pr)
Fiber2 = Passivefiber_class.from_data_sheet(fiberdata_path,
                                            'OFS_TruewaveXL.json',
                                            lam_pr)


# Define numerical parameters
Tmax = T0*4             # Simulation window size (ns)
N = 2**12
t = np.linspace(-Tmax/2,Tmax/2,N)
Nz_save = 21
Nsec = 1

L1_vec = np.arange(1,74,10)*1e3
L2_vec = L-L1_vec

N_sweep = len(L1_vec)

# %% Run simulation
# Insert directory for saving data
savedir = this_dir+r'\data\MI_test\altfiber_sec3\P150'

def sim_func(args):
    i, Ppeak0, t, T0, L1, L2, Nz_save, Fiber1, Fiber2, PSDnoise_dbmGHz, Nsec, savedir = args
    print('Start:\t Simulation no. '+str(i))
    start_time = time.time()
    A0 = A0_func(t, T0, Ppeak0)
    A0 = np.sqrt(Ppeak0)*np.ones(len(t))
    S = Simulation_pulsed_sections2_fiber(t, A0, L1, L2, Nz_save, Fiber1,
                                          Fiber2, PSDnoise_dbmGHz, Nsec)
    z, A = S.run()
    #savefname = f'P0_{int(Ppeak0 * 1000)}.pkl'
    savefname = f'L_{int(L1*1e-3)}.pkl'
    S.save_pickle(savedir, savefname)
    end_time = time.time()
    print('End:\t Simulation no. '+str(i)+' Time: '+str(end_time-start_time))

if __name__ == '__main__':
    args_list = [(i, Ppeak0, t, T0, L1_vec[i],L2_vec[i], Nz_save, Fiber1,Fiber2, 
                  PSDnoise_dbmGHz, Nsec, savedir) for i in range(N_sweep)]
    #for args in args_list:
    #    sim_func(args)
    with multiprocessing.Pool() as pool:
        pool.map(sim_func, args_list)
                 

# %%


