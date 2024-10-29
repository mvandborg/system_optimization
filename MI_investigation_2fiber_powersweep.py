
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
from src.help_functions import PSD_dbmGHz2dbmnm, PSD_dbmnm2dbmGHz,dbm
from MI_investigation_sweep import get_ASE_noise_WGHz

# %% Define propagation fibers
def A0_func(t,T0,Ppeak0):
    return sqrt(Ppeak0)*exp(-(2*t/T0)**22)

# Define physical parameters
L = 100e3                # Fiber length (km)
T0 = 2                # Pulse length (ns)

lam_pr = 1550e-9
Ppeak0 = 100e-3

PSD_ase = get_ASE_noise_WGHz(G=100,lam0=lam_pr)
PSD_ase_dbmGHz = dbm(PSD_ase)
PSDnoise_dbmGHz = PSD_ase_dbmGHz
PSD_noise_dbmnm = PSD_dbmGHz2dbmnm(PSDnoise_dbmGHz,lam_pr*1e9,2.998e8)
dnu = 1e-3      # Linewidth of the laser (GHz)

fiberdata_path = os.path.join(this_dir, 'fiber_data')
Fiber1 = Passivefiber_class.from_data_sheet(fiberdata_path,
                                            'OFS_TruewaveXL.json',
                                            lam_pr)
Fiber2 = Passivefiber_class.from_data_sheet(fiberdata_path,
                                            'OFS_SCUBA150.json',
                                            lam_pr)
# change losses to fit from experiments
#Fiber1.alpha = 0.166/4.34/1e3
#Fiber2.alpha = 0.197/4.34/1e3

# Define numerical parameters
Tmax = T0*4             # Simulation window size (ns)
N = 2**12
t = np.linspace(-Tmax/2,Tmax/2,N)
Nz_save = 11
Nsec = 3

L1 = 30e3
L2 = 75e3-L1

step_sweep = 20
#Ppeak0_vec = np.array([165,192,231,262,301,368])*1e-3
#Ppeak0_vec = np.linspace(0.8*165,1.2*368,41)*1e-3
Ppeak0_vec = np.arange(40,100,step_sweep)*1e-3
N_sweep = len(Ppeak0_vec)

# %% Run simulation
# Insert directory for saving data
savedir = this_dir+r'\data\MI_test\altfiber_sec3\L30'

def sim_func(args):
    i, Ppeak0, t, T0, L1, L2, Nz_save, Fiber1, Fiber2, \
        PSDnoise_dbmGHz, Nsec, dnu, savedir = args
    print('Start:\t Simulation no. '+str(i))
    start_time = time.time()
    A0 = A0_func(t, T0, Ppeak0)
    A0 = np.sqrt(Ppeak0)*np.ones(len(t))
    S = Simulation_pulsed_sections2_fiber(t, A0, L1, L2, Nz_save, Fiber1,
                                          Fiber2, PSDnoise_dbmGHz, Nsec,
                                          linewidth=dnu)
    z, A = S.run()
    savefname = f'P0_{int(Ppeak0 * 1000)}.pkl'
    #savefname = f'L_{int(L1*1e-3)}.pkl'
    S.save_pickle(savedir, savefname)
    end_time = time.time()
    print('End:\t Simulation no. '+str(i)+' Time: '+str(end_time-start_time))

if __name__ == '__main__':
    args_list = [(i, Ppeak0_vec[i], t, T0, L1,L2, Nz_save,Fiber1,Fiber2, 
                  PSDnoise_dbmGHz, Nsec, dnu, 
                  savedir) for i in range(N_sweep)]
    #for args in args_list:
    #    sim_func(args)
    with multiprocessing.Pool() as pool:
        pool.map(sim_func, args_list)
                 

# %%


