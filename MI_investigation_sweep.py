
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
from src.simulation_system import Simulation_pulsed_sections_fiber
from src.help_functions import PSD_dbmGHz2dbmnm, PSD_dbmnm2dbmGHz

# %% Define propagation fibers
def A0_func(t,T0,Ppeak0):
    return sqrt(Ppeak0)*exp(-(2*t/T0)**52)

# Directory for saving the data

savedir = os.path.join(this_dir,'data/MI_test/sec1_SumULL')

L = 100e3               # Fiber length (km)
T0 = 100                # Pulse length (ns)
lam_p = 1455e-9         # Wavelength (m)
lam_pr = 1550e-9
lam_arr = np.array([lam_p,lam_pr])
Ppeak0 = 100e-3
PSD_noise_dbmnm = -30
PSDnoise_dbmGHz = PSD_dbmnm2dbmGHz(PSD_noise_dbmnm,lam_pr*1e9,2.998e8)

fiberdata_path = os.path.join(this_dir, 'fiber_data')
Fiber = Passivefiber_class.from_data_sheet( fiberdata_path,
                                            'Sumitomo_ULL.json',
                                            lam_arr)

Tmax = T0*5             # Simulation window size (ns)
N = 2**16
t = np.linspace(-Tmax/2,Tmax/2,N)
Nz_save = 11
Nsec = 1

step_sweep = 10
Ppeak0_vec = np.arange(10,350,step_sweep)*1e-3      # Pump power (W)
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
    savefname = f'P0_{int(Ppeak0 * 1000)}.pkl'
    #savefname = f'P0_{int(-PSDnoise_dbmGHz)}.pkl'
    S.save_pickle(savedir, savefname)
    end_time = time.time()
    print('End:\t Simulation no. '+str(i)+' Time: '+str(end_time-start_time))

if __name__ == '__main__':
    args_list = [(i, Ppeak0_vec[i], t, T0, L, Nz_save, Fiber, 
                  PSDnoise_dbmGHz, Nsec, savedir) for i in range(N_sweep)]
    with multiprocessing.Pool() as pool:
        pool.map(sim_func, args_list)
                 

# %%

