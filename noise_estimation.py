

# This program estimates the noise in the Lios Interrogator 
# The program calculates the thermal noise / dark current shot noise contributions and the ASE noise

# %%% Import data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def db(x):
    return 10*np.log10(x)

filedir = r"C:\Users\madshv\OneDrive - Danmarks Tekniske Universitet\data_analysis\lios_noise_estimation_data"

# %% Estimation of thermal noise and dark current shot noise

filedir_ext = r'/150kmNoErb/'
filename = r'150kmMax_5kPCspec.csv'
filename_pow = r'150kmMax_5kPCPow.csv'

# ne meaning 'no erbium'
df_ne = pd.read_csv(filedir+filedir_ext+filename,header=None,names=['z','f0'])
df_ne_pow = pd.read_csv(filedir+filedir_ext+filename_pow,header=None,names=['z','Pb'])

z_ne = df_ne['z'].to_numpy()
f0_ne = df_ne['f0'].to_numpy()
N_ne = len(z_ne)

z_ne_pow = df_ne_pow['z'].to_numpy()
Pb_ne_pow = df_ne_pow['Pb'].to_numpy()

# Calculate the moving standard deviation
std_ne = np.zeros(N_ne)
std_ne_pow = np.zeros(N_ne)
win_size = 40
idx_start = win_size+50
for i in range(idx_start,N_ne):
    i_win_min = 0
    i_win_max = N_ne-1
    if i>win_size:
        i_win_min = i-win_size
    if i<N_ne-1-win_size:
        i_win_max = i+win_size
    std_ne[i] = np.std(f0_ne[i_win_min:i_win_max])
    std_ne_pow[i] = np.std(Pb_ne_pow[i_win_min:i_win_max])

# Fit noise model
idx_sec1 = (z_ne>z_ne[idx_start])*(z_ne<95e3)
idx_sec2 = (z_ne>100e3)*(z_ne<145e3)
C1 = np.mean(std_ne[idx_sec1])
C2_log = np.polyfit(z_ne[idx_sec2],np.log(std_ne[idx_sec2]),1)

C2 = np.exp(C2_log[1])
alpha_fit = C2_log[0]/2

plt.close('all')
fig0,ax0 = plt.subplots(2,1,constrained_layout=True)
ax0[0].plot(z_ne*1e-3,f0_ne)
ax0[0].set_ylim([10950,11050])
ax0[1].plot(z_ne*1e-3,db(std_ne))
ax0[1].plot(z_ne[idx_sec1]*1e-3,db(C1*z_ne[idx_sec1]**0))
ax0[1].plot(z_ne[idx_sec2]*1e-3,db(C2*np.exp(2*alpha_fit*z_ne[idx_sec2])))
ax0[1].plot(z_ne*1e-3,db(C1+C2*np.exp(2*alpha_fit*z_ne)))
ax0[1].set_ylim([-10,20])

fig1,ax1 = plt.subplots(2,1,constrained_layout=True)
ax1[0].plot(z_ne_pow*1e-3,-db(np.abs(Pb_ne_pow)))
ax1[1].plot(z_ne_pow*1e-3,db(std_ne_pow))
ax1[0].set_ylim([-20,-5])
ax1[1].set_ylim([-15,15])

# %%
