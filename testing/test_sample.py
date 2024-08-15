

import sys
import os
# Add the parent directory to the sys.path
this_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
parent_dir = os.path.abspath(os.path.join(this_dir, '..'))
sys.path.append(parent_dir)

import src.fiberdata_passive as fibdat
from src.simulation_system import Simulation_pulsed_sections_fiber
import numpy as np
from numpy import sqrt,exp
import pickle

# Test if the data is imported correctly by the fiberdata class
def test_fiberdata_import():
    lam = 1550e-9
    Fiber = fibdat.Passivefiber_class.from_data_sheet(parent_dir+'/testing/data','test_fiber_data.json',lam)
    
    # Correct data values
    beta2 = -2.168207392672026e-08
    Aeff = 85e-12
    gamma = 0.00123995
    omega = 1215289.6484467355
    
    # Check data values
    assert abs((Fiber.Aeff[0]-Aeff)/Aeff)<1e-3
    assert abs((Fiber.beta2-beta2)/beta2)<1e-3
    assert abs((Fiber.gamma[0]-gamma)/gamma)<1e-3
    assert abs((Fiber.omega-omega)/omega)<1e-3

# Test if the nonlinear schrodinger equation produces the correct output
def test_NLSE():
    
    # Load reference data
    with open(os.path.join(this_dir,'data/simulation_data.pkl'), "rb") as pkl_file:
        data = pickle.load(pkl_file)
        Fiber_dict = data['Fiber_dict']
        t = data['t']
        f = data['f']
        z = data['z']
        A = data['A']
        PSDnoise_dbmHz = data['PSDnoise_dbmHz']
        L = data['L']
        dnu = data['linewidth']
        
    PSDnoise_dbmGHz = PSDnoise_dbmHz+90
    Nz = len(z)
    lam_pr = Fiber_dict['lam']
    
    Nz_save = len(z)
    Nsec = 1
    
    Fiber = fibdat.Passivefiber_class.from_data_sheet( this_dir,
                                                'data/test_fiber_data.json',
                                                lam_pr)
    
    A0 = A[:,0]
    
    # Simulate data
    S = Simulation_pulsed_sections_fiber(t, A0, L, Nz_save, Fiber, Nsec,
                                        PSDnoise_dbmGHz, linewidth=dnu)
    z_sim, A_sim = S.run()
    
    # Compare
    assert np.sum((z_sim-z)**2)<1e-3
    assert np.sum((A_sim.real-A.real)**2)<1e-3
    assert np.sum((A_sim.imag-A.imag)**2)<1e-3
    
    
    












