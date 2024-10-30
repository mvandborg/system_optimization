
# %% Load modules

import os
import sys
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

import numpy as np
import matplotlib.pyplot as plt
from plotting_MInoise import SignalAnalyzer,sort_res
from src.help_functions import db,dbm,PSD_dbmGHz2dbmnm
from scipy.constants import c
c = c*1e-9      # Unit m/ns

def plot_SNR_vs_power(param_list_vec,R_vec):
    fig,ax = plt.subplots(constrained_layout=True)
    for i in range(len(R_vec)):
        R = R_vec[i]
        param_vec = param_list_vec[i]
        y = [r.SNR[-1] for r in R]  # Unit W/Hz
        lab = '$L_1$={:.0f} km'.format(R[0].L[0]*1e-3)
        ax.plot(param_vec,dbm(y),label=lab)
    ax.grid()
    ax.legend()
    ax.set_xlabel('$P_{pr}(0)$ (mW)')
    ax.set_ylabel(r'SNR')


# %% Import and process data
if __name__ =='__main__':
    relative_dir = "data/MI_test/altfiber_sec3"

    data_dir_vec = os.listdir(file_dir+'/'+relative_dir)
    subfolder_path_vec = [file_dir+'/'+relative_dir+'/'+data_dir \
        for data_dir in data_dir_vec]

    file_list_vec = [os.listdir(subpath) for subpath in subfolder_path_vec]
    Nfile_vec = [len(f) for f in file_list_vec]
    Nsubfold = len(subfolder_path_vec)

    R_vec = []
    param_list_vec = []

    for i in range(Nsubfold):
        R1 = []
        param_list = []
        file_list = file_list_vec[i]
        subfolder_path = subfolder_path_vec[i]
        for file_name in file_list:
            S = SignalAnalyzer(subfolder_path,file_name)
            R1.append(S)
            param_list.append(S.param)
            param_list,R1 = sort_res(param_list,R1)
        param_list_vec.append(param_list)
        R_vec.append(R1)
    
# %% Plotting
if __name__ == '__main__':
    plt.close('all')
    plot_SNR_vs_power(param_list_vec,R_vec)
    plt.show()
    
    
# %%
