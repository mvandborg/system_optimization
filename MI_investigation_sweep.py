
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
from src.help_functions import PSD_dbmGHz2dbmnm, PSD_dbmnm2dbmGHz, dbm,\
    get_ASE_noise_WGHz

# Converting a peak power to a gain in the setup at DFM
def convert_Ppr2gain(Ppr):
    # Ppr: Probe peak power in W
    A0 = 3.89e3
    A1 = -58.54
    return A0*Ppr+A1

# %% Define propagation fibers
def A0_func(t,T0,Ppeak0):
    return sqrt(Ppeak0)*exp(-(2*t/T0)**52)

if __name__ == '__main__':
    # Directory for saving the data
    savedir = os.path.join(
            this_dir,
            'data\\MI_test\\meas_compare_dfm\\noise_calculated_fine'
            )

    L = 100e3               # Fiber length (km)
    T0 = 5                 # Pulse length (ns)
    lam_pr = 1550e-9
    Ppeak0 = 300e-3         

    # Linewidth of laser
    dnu = 1e-5      # Linewidth of the laser (GHz)
    # DFM: 10 kHz (1e-5)
    # Lios: 1 MHz (1e-3)

    fiberdata_path = os.path.join(this_dir, 'fiber_data')
    Fiber = Passivefiber_class.from_data_sheet( fiberdata_path,
                                                'SMF28_generic.json',
                                                lam_pr)

    Tmax = T0*4             # Simulation window size (ns)
    N = 2**13
    t = np.linspace(-Tmax/2,Tmax/2,N)
    Nz_save = 21
    Nsec = 1

    step_sweep = 20
    #Ppeak0_vec = np.array([166,192,231,263,301,369])*1e-3 
    #Ppeak0_vec = np.array([168,188,211,236,264,297,334,374,
    #                       420,471,529,592,664])*1e-3
    #Ppeak0_vec = np.array([134,149,166,184,205,230,256,285,317,
    #                       355,398,444,496])*1e-3    
    #Ppeak0_vec = np.array([165,192,231,262,301,368])*1e-3

    #Ppeak0_vec = np.arange(460,550,step_sweep)*1e-3      # Pump power (W)
    #PSDnoise_dbmGHz_vec = np.array([-51,-46,-41,-36,-31,-26])
    
    # For the new measurements
    Ppeak0_vec = np.array([138,158,174,203,232,254,279,
                           321,357,414,448,512,564])*1e-3
    Ppeak0_vec = np.linspace(110,620,51)*1e-3      # Pump power (W)
    Ppeak0_vec = np.array([Ppeak0_vec[-2]])
    G_vec = convert_Ppr2gain(Ppeak0_vec)
        
    # Noise calculations
    PSD_ase = get_ASE_noise_WGHz(G=G_vec,lam0=lam_pr)
    PSD_ase_dbmGHz = dbm(PSD_ase)
    PSDnoise_dbmGHz = PSD_ase_dbmGHz
    PSD_noise_dbmnm = PSD_dbmGHz2dbmnm(PSDnoise_dbmGHz,lam_pr*1e9,2.998e8)
    
    N_sweep = len(Ppeak0_vec)

# %% Run simulation
def sim_func(args):
    i, Ppeak0, t, T0, L, Nz_save, Fiber, Nsec, \
        PSDnoise_dbmGHz, dnu, savedir = args
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
                  Nsec, PSDnoise_dbmGHz[i], 
                  dnu, savedir) for i in range(N_sweep)]
    with multiprocessing.Pool() as pool:
        pool.map(sim_func, args_list)
    #for arg in args_list: 
    #    sim_func(arg)
                 

# %%

