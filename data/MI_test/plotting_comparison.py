
# %% Load modules
import os
import sys
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
sys.path.insert(0, 'C:/Users/madshv/OneDrive - Danmarks Tekniske Universitet/code')
import numpy as np
import matplotlib.pyplot as plt
from data.MI_test.plotting import SignalAnalyzer,sort_res
from help_functions import db,dbm
from scipy.constants import c
c = c*1e-9

# %% Import and process data


subfolder_path1 = r"C:\Users\madshv\OneDrive - Danmarks Tekniske Universitet\code\system_optimization\data\MI_test\sec1"
subfolder_path2 = r"C:\Users\madshv\OneDrive - Danmarks Tekniske Universitet\code\system_optimization\data\MI_test\sec2"
subfolder_path3 = r"C:\Users\madshv\OneDrive - Danmarks Tekniske Universitet\code\system_optimization\data\MI_test\sec3"

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
fig0,ax0 = plt.subplots(1,2,constrained_layout=True)
for i in range(Nsubfold):
    R = R_vec[i]
    param_vec = param_list_vec[i]
    y = db([r.PSDbril_PSDmi_ratio[-1] for r in R])
    ax0[0].plot(param_vec,y)
    ax0[0].set_xlabel('P0 (mW)')
    ax0[0].set_ylabel(r'$PSD_{bril}/PSD_{ray}$ (dB)')
    y = dbm([r.P_inband[-2] for r in R])
    ax0[1].plot(param_vec,y)
    ax0[1].set_xlabel(r'$P_0$')
    ax0[1].set_ylabel(r'$P_{inband}$ (dBm)')
# %%
