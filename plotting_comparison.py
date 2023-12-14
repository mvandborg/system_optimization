
# %% Load modules

import os
import sys
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

import numpy as np
import matplotlib.pyplot as plt
from plotting import SignalAnalyzer,sort_res
from src.help_functions import db,dbm,PSD_dbmGHz2dbmnm
from scipy.constants import c
c = c*1e-9                      # Unit m/ns

# %% Import and process data

#subfolder_path1 = r"C:\Users\madshv\OneDrive - Danmarks Tekniske Universitet\code\system_optimization\data\MI_test\sec1"
#subfolder_path2 = r"C:\Users\madshv\OneDrive - Danmarks Tekniske Universitet\code\system_optimization\data\MI_test\sec2"
#subfolder_path3 = r"C:\Users\madshv\OneDrive - Danmarks Tekniske Universitet\code\system_optimization\data\MI_test\sec3"

#subfolder_path1 = r"C:\Users\madshv\OneDrive - Danmarks Tekniske Universitet\code\system_optimization\data\MI_test\sec1\noise_-151"
#subfolder_path2 = r"C:\Users\madshv\OneDrive - Danmarks Tekniske Universitet\code\system_optimization\data\MI_test\sec1\noise_-141"
#subfolder_path3 = r"C:\Users\madshv\OneDrive - Danmarks Tekniske Universitet\code\system_optimization\data\MI_test\sec1\noise_-131"
#subfolder_path4 = r"C:\Users\madshv\OneDrive - Danmarks Tekniske Universitet\code\system_optimization\data\MI_test\sec1\noise_-121"

subfolder_path1 = r"C:\Users\madshv\OneDrive - Danmarks Tekniske Universitet\code\system_optimization\data\MI_test\altfiber_sec3\P100"
subfolder_path2 = r"C:\Users\madshv\OneDrive - Danmarks Tekniske Universitet\code\system_optimization\data\MI_test\altfiber_sec3\P130"
subfolder_path3 = r"C:\Users\madshv\OneDrive - Danmarks Tekniske Universitet\code\system_optimization\data\MI_test\altfiber_sec3\P150"


subfolder_path_vec = [subfolder_path1,subfolder_path2,subfolder_path3]

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
plt.close('all')

fig0,ax0 = plt.subplots(1,3,constrained_layout=True)
for i in range(Nsubfold):
    R = R_vec[i]
    param_vec = param_list_vec[i]
    y = [r.PSD_rayscat[-1] for r in R]  # Unit W/Hz
    #lab = PSD_dbmGHz2dbmnm(float(subfolder_path_vec[i][-4:]) + 90, 1550, 3e8)
    #lab = "Pnoise= {:.1f} dBm/nm".format(lab)
    lab = 'test'
    ax0[0].plot(param_vec,dbm(y),label=lab)
    y = [r.PSD_bril[-1]/r.PSD_rayscat[-1] for r in R]
    ax0[1].plot(param_vec,db(y),label=lab)
    y = [r.PSD_bril[-1] for r in R]     # Unit W/Hz
    ax0[2].plot(param_vec,dbm(y),label=lab)
ax0[0].grid()
ax0[0].legend()
ax0[0].set_xlabel('P0 (mW)')
ax0[0].set_ylabel(r'$PSD_{ray}$ (dBm/Hz)')
ax0[1].grid()
ax0[1].legend()
ax0[1].set_xlabel(r'$P_0$ (mW)')
ax0[1].set_ylabel(r'$PSD_{bril}/PSD_{ray}$ (dB)')
ax0[2].grid()
ax0[2].legend()
ax0[2].set_xlabel(r'$P_0$ (mW)')
ax0[2].set_ylabel(r'$PSD_{bril}$ (dBm/Hz)')



kvec = np.array([1e-10])
fig1,ax1 = plt.subplots(constrained_layout=True)
for isec in range(Nsubfold):
    for ik in range(len(kvec)):
        k = kvec[ik]
        y = db([1/(1/r.PSDbril_PSDmi_ratio[-1]+k/r.PSD_bril[-1]) for r in R_vec[isec]])
        ax1.plot(param_list_vec[isec],y,label=f'{isec} Sections')
        #ax1[isec].set_title(f'Sec {subfolder_path_vec[isec][-1]}')
        #lab = PSD_dbmGHz2dbmnm(float(subfolder_path_vec[isec][-4:]) + 90, 1550, 3e8)
        #lab = "Pnoise= {:.1f} dBm/nm".format(lab)
        lab = 'test'
        ax1.set_title(lab)
        ax1.set_xlim([0,410])
    ax1.grid()
    ax1.set_xlabel('P0 (mW)')
    ax1.set_ylabel(r'SNR (dB)')
    ax1.legend()
plt.show()

# %%
