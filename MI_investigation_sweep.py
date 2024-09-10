
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
from src.help_functions import PSD_dbmGHz2dbmnm, PSD_dbmnm2dbmGHz, dbm

# %% Define propagation fibers
def A0_func(t,T0,Ppeak0):
    return sqrt(Ppeak0)*exp(-(2*t/T0)**52)

def get_ASE_noise_WGHz(G=20,lam0=1550e-9):
    n_sp = 2
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

# Directory for saving the data

savedir = os.path.join(this_dir,'data\\MI_test\\meas_compare_dfm\\noise_-15_fitpower')

L = 84e3               # Fiber length (km)
T0 = 10                # Pulse length (ns)
lam_pr = 1540e-9
Ppeak0 = 100e-3

# EDFA noise calculation

PSD_ase = get_ASE_noise_WGHz(G=1e3,lam0=lam_pr)
PSD_ase_dbmGHz = dbm(PSD_ase)
PSDnoise_dbmGHz = PSD_ase_dbmGHz
PSD_noise_dbmnm = PSD_dbmGHz2dbmnm(PSDnoise_dbmGHz,lam_pr*1e9,2.998e8)
dnu = 10e-6      # Linewidth of the laser (GHz)

fiberdata_path = os.path.join(this_dir, 'fiber_data')
Fiber = Passivefiber_class.from_data_sheet( fiberdata_path,
                                            'Corning_SMF28.json',
                                            lam_pr)

Tmax = T0*7             # Simulation window size (ns)
N = 2**14
t = np.linspace(-Tmax/2,Tmax/2,N)
Nz_save = 21
Nsec = 1

step_sweep = 10
#Ppeak0_vec = np.array([166,192,231,263,301,369])*1e-3 
Ppeak0_vec = np.array([168,188,211,236,264,297,334,374,
                       420,471,529,592,664])*1e-3 
Ppeak0_vec = np.array([p/2 for p in Ppeak0_vec])
#Ppeak0_vec = np.arange(10,250,step_sweep)*1e-3      # Pump power (W)
#PSDnoise_dbmGHz_vec = np.array([-151,-141,-131,-121])+90
N_sweep = len(Ppeak0_vec)

# %% Run simulation
def sim_func(args):
    i, Ppeak0, t, T0, L, Nz_save, Fiber, Nsec, PSDnoise_dbmGHz, dnu, savedir = args
    print('Start:\t Simulation no. '+str(i))
    start_time = time.time()
    A0 = A0_func(t, T0, Ppeak0)
    A0 = np.sqrt(Ppeak0)*np.ones(len(t))
    S = Simulation_pulsed_sections_fiber(t, A0, L, Nz_save, Fiber, Nsec,
                                         PSDnoise_dbmGHz=PSDnoise_dbmGHz, 
                                         linewidth=dnu)
    z, A = S.run()
    savefname = f'P0_{int(Ppeak0 * 1000)}.pkl'
    #savefname = f'P0_{int(-PSDnoise_dbmGHz)}.pkl'
    S.save_pickle(savedir, savefname)
    end_time = time.time()
    print('End:\t Simulation no. '+str(i)+' Time: '+str(end_time-start_time))

if __name__ == '__main__':
    args_list = [(i, Ppeak0_vec[i], t, T0, L, Nz_save, Fiber,
                  Nsec, PSDnoise_dbmGHz, dnu, savedir) for i in range(N_sweep)]
    with multiprocessing.Pool() as pool:
        pool.map(sim_func, args_list)
    #for arg in args_list: 
    #    sim_func(arg)
                 

# %%

