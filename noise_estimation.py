

# This program estimates the noise in the Lios Interrogator 
# The program calculates the thermal noise / dark current shot noise contributions and the ASE noise

# %%% Import data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.fft import fft,fftfreq,fftshift

def db(x):
    return 10*np.log10(x)

def invdb(x):
    return 10**(x/10)

def std_moving(x,idx_start,win_size):
    N = len(x)
    stdx = np.zeros(N)
    meanx = np.zeros(N)
    for i in range(idx_start,N):
        i_win_min = 0
        i_win_max = N-1
        if i>win_size:
            i_win_min = i-win_size
        if i<N-1-win_size:
            i_win_max = i+win_size
        stdx[i] = np.std(x[i_win_min:i_win_max])
        meanx[i] = np.mean(x[i_win_min:i_win_max])
    return meanx,stdx

def f_model(z,c1,c2,c3):
    return c1+c2*np.exp(0.16*2e-3/4.34*z)
    #return c1+c2*np.exp(c3*z)

filedir = r"C:\Users\madshv\OneDrive - Danmarks Tekniske Universitet\data_analysis\lios_noise_estimation_data"

# %% Estimation of thermal noise and dark current shot noise

filedir_ext = r'/150kmNoErb/'
filename_f0 = r'150kmMax_5kPCspec.csv'
filename_pow = r'150kmMax_5kPCPow.csv'

# ne meaning 'no erbium'
df_ne = pd.read_csv(filedir+filedir_ext+filename_f0,header=None,names=['z','f0'])
df_ne_pow = pd.read_csv(filedir+filedir_ext+filename_pow,header=None,names=['z','Pb'])

z_ne = df_ne['z'].to_numpy()
f0_ne = df_ne['f0'].to_numpy()
N_ne = len(z_ne)

Pb_ne = invdb(df_ne_pow['Pb'].to_numpy())

# Calculate the moving standard deviation
win_size = 25
idx_start = win_size+50
mean_f0_ne,std_f0_ne = std_moving(f0_ne,idx_start,win_size)
mean_pow_ne,std_pow_ne = std_moving(Pb_ne,idx_start,win_size)

SNR_ne = mean_pow_ne/std_pow_ne
inv_SNR_ne = 1/SNR_ne

# Fit noise model
idx_sec1 = (z_ne>z_ne[idx_start])*(z_ne<80e3)*(z_ne>60e3)
idx_sec2 = (z_ne>105e3)*(z_ne<140e3)
C1_guess = np.mean(std_f0_ne[idx_sec1])
C2_log = np.polyfit(z_ne[idx_sec2],np.log(std_f0_ne[idx_sec2]),1)
C2_guess = np.exp(C2_log[1])
C3_guess = C2_log[0]

idxfit_ne = (z_ne>40e3)*(z_ne<145e3)
fit_f0_e = curve_fit(f_model,z_ne[idxfit_ne],std_f0_ne[idxfit_ne],p0=[C1_guess,C2_guess,C3_guess],maxfev=5000)
C1,C2,C3 = fit_f0_e[0]


D1_guess = np.mean(inv_SNR_ne[idx_sec1])
D2_log = np.polyfit(z_ne[idx_sec2],np.log(inv_SNR_ne[idx_sec2]),1)
D2_guess = np.exp(D2_log[1])
D3_guess = D2_log[0]

fit_invsnr_e = curve_fit(f_model,z_ne[idxfit_ne],inv_SNR_ne[idxfit_ne],p0=[D1_guess,D2_guess,D3_guess],maxfev=5000)
D1,D2,D3 = fit_invsnr_e[0]

# Plots
plt.close('all')
fig0,ax0 = plt.subplots(2,1,constrained_layout=True)
ax0[0].plot(z_ne*1e-3,f0_ne)
ax0[0].set_ylim([10950,11050])
ax0[0].set_xlabel('z (km)')
ax0[0].set_ylabel('Peak frequency (MHz)')
ax0[1].semilogy(z_ne*1e-3,std_f0_ne,label='Meas')
ax0[1].semilogy(z_ne[idxfit_ne]*1e-3,f_model(z_ne[idxfit_ne],C1,C2,C3),label='Model fit 2')
ax0[1].set_ylim([0.2,20])
ax0[1].set_xlabel('z (km)')
ax0[1].set_ylabel('Standard deviation (MHz)')
ax0[1].legend()

fig1,ax1 = plt.subplots(2,1,constrained_layout=True)
ax1[0].plot(z_ne*1e-3,db(Pb_ne))
ax1[0].set_ylim([-55,0])
ax1[0].set_xlabel('z (km)')
ax1[0].set_ylabel('Peak power (dB)')
ax1[1].semilogy(z_ne*1e-3,inv_SNR_ne,label='Meas')
ax1[1].semilogy(z_ne[idxfit_ne]*1e-3,f_model(z_ne[idxfit_ne],D1,D2,D3),label='Model fit')
ax1[1].set_ylim([8e-3,1e0])
ax1[1].set_xlabel('z (km)')
ax1[1].set_ylabel(r'$1/SNR=std(P_b)/P_b$ (MHz)')
ax1[1].legend()

idxfft = (z_ne>10e3)*(z_ne<40e3)
f0_ne_fft = f0_ne[idxfft]
fig2,ax2 = plt.subplots(1,2,constrained_layout=True)
ax2[0].plot(z_ne*1e-3,f0_ne)
ax2[0].set_xlabel('z (m)')
ax2[0].set_ylabel('Peak frequency (MHz)')
ax2[0].set_ylim([10960,11020])
ax2[1].semilogy(fftshift(fftfreq(len(f0_ne_fft),d=z_ne[1]-z_ne[0])),
                fftshift(np.abs(fft(f0_ne_fft))))

# %% Estimation of ASE noise from erbium amplifier

filedir_ext = r'/200km/200kmWithJesperCode/'

expname = ['20','19_5','19','18_5','18','17_5','17']
dat_pump_e = pd.read_csv(filedir+filedir_ext+'powers.csv')

# Convert attenuations from strings to float
Att = []
for strp in expname:
    Att_tmp = float(strp[0:2])
    if len(strp)>2:
        Att_tmp+=float(strp[-1])/10
    Att.append( Att_tmp )
Att = np.array(Att)

# e meaning erbium
f0_e = []
Pb_e = []
N_exp = len(expname)
for i in range(N_exp):
    filename_pow = 'Att'+expname[i]+'Pow.csv'
    filename_f0 = 'Att'+expname[i]+'spec.csv'
    df_pow_e = pd.read_csv(filedir+filedir_ext+filename_pow,header=None,names=['z','Pb'])
    df_f0_e = pd.read_csv(filedir+filedir_ext+filename_f0,header=None,names=['z','f0'])
    f0_e.append(df_f0_e['f0'].to_numpy())
    Pb_e_db = df_pow_e['Pb'].to_numpy()
    Pb_e.append(invdb(Pb_e_db))
z_e = df_pow_e['z']

# Translate arrenuation into powers
Pp = []
for i in range(N_exp):
    dAtt = np.abs(dat_pump_e['Attenuation'].to_numpy()-Att[i])
    idx_Pp = np.argmin(dAtt)
    if dAtt[idx_Pp]<1e-3:
        Pp.append( dat_pump_e['Power (mW)'][idx_Pp])
Pp = np.array(Pp)


# %% Post processing

# Calculate standard deviation of power and peak frequency
N_e = len(z_e)
win_size = 50
idx_start = win_size+50
std_f0_e = []
std_pow_e = []
invSNR_pow_e = []
for i in range(N_exp):
    print(i)
    mean_f0_e_tmp,std_f0_e_tmp = std_moving(f0_e[i],idx_start,win_size)
    std_f0_e.append( std_f0_e_tmp )
    mean_pow_e_tmp,std_pow_e_tmp = std_moving(Pb_e[i],idx_start,win_size)
    std_pow_e.append( std_pow_e_tmp )
    invSNR_pow_e.append( std_pow_e_tmp/mean_pow_e_tmp )

# Fit the standard deviation data with f(x)=c1+c2*exp(2*alpha*z)
idx_sec1_e = (z_e>z_e[idx_start])*(z_e<48e3)
idx_sec2_e = (z_e>60e3)*(z_e<95e3)
idx_fit_e = (z_e>z_e[idx_start])*(z_e<96e3)*(z_e>0e3)

C1_e = np.zeros(N_exp)
C2_e = np.zeros(N_exp)
C3_e = np.zeros(N_exp)
D1_e = np.zeros(N_exp)
D2_e = np.zeros(N_exp)
D3_e = np.zeros(N_exp)
for i in range(N_exp):
    print(i)
    C1_guess = np.mean(std_f0_e[i][idx_sec1_e])
    C2_log = np.polyfit(z_e[idx_sec2_e],np.log(std_f0_e[i][idx_sec2_e]),1)
    C2_guess = np.exp(C2_log[1])
    C3_guess = C2_log[0]
    fit_f0_e = curve_fit(f_model,z_e[idx_fit_e],std_f0_e[i][idx_fit_e],p0=[C1_guess,C2_guess,C3_guess],maxfev=5000)
    C1_e[i],C2_e[i],C3_e[i] = fit_f0_e[0]

    D1_guess = np.mean(invSNR_pow_e[i][idx_sec1_e])
    D2_log = np.polyfit(z_e[idx_sec2_e],np.log(invSNR_pow_e[i][idx_sec2_e]),1)
    D2_guess = np.exp(D2_log[1])
    D3_guess = D2_log[0]
    fit_invsnr_e = curve_fit(f_model,z_e[idx_fit_e],invSNR_pow_e[i][idx_fit_e],p0=[D1_guess,D2_guess,D3_guess],maxfev=5000)
    D1_e[i],D2_e[i],D3_e[i] = fit_invsnr_e[0]

alpha_fit_f0_e = C3_e*4.34/2e-3
alpha_fit_Pb_e = D3_e*4.34/2e-3

# Simulation

data_sim = np.load(filedir+'/200km/simulation_asenoise.npy',allow_pickle=True)
Nsim = len(data_sim[0])

BWnoise = data_sim[6]
lamnoise = data_sim[3]
idxlamnoise= np.argmin(lamnoise*1e6<1.55)
Sase_bw = []
Sase_bw_1550 = []
for i in range(Nsim):
    Sase_bw_tmp = data_sim[5][i][:,0]/BWnoise*np.exp(-0.16/4.34*100)
    Sase_bw.append( Sase_bw_tmp )
    Sase_bw_1550.append( Sase_bw_tmp[idxlamnoise]*1e18)
Sase_bw_1550 = np.array(Sase_bw_1550)

# Estimation of ase noise contribution
# We fit to the model f(z)=C1+(C20+C21*Sase)*exp(c3*z), c3 = -2*alpha

C21_e,C20_e = np.polyfit(Sase_bw_1550,C2_e,1)
D21_e,D20_e = np.polyfit(Sase_bw_1550,D2_e,1)

# %%
plt.close('all')
fig2,ax2 = plt.subplots(2,1,constrained_layout=True)
colarr = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink']
idxplot = np.arange(0,N_e-1,1)
for j in range(N_exp):
    i = N_exp-j-1
    ax2[0].semilogy(z_e[idxplot]*1e-3,f0_e[i][idxplot])
    ax2[1].semilogy(z_e[idxplot]*1e-3,std_f0_e[i][idxplot],label=str(Att[i]),color=colarr[i])
    ax2[1].semilogy(z_e[idx_fit_e]*1e-3,f_model(z_e[idx_fit_e],C1_e[i],C2_e[i],C3_e[i]),
                    label='Model fit',ls='--',color='black')
ax2[0].set_ylim([10960,11010])
ax2[1].set_ylim([0.3,10])
ax2[0].set_xlabel('z (km)')
ax2[1].set_xlabel('z (km)')
ax2[0].set_ylabel('Peak frequency (MHz)')
ax2[1].set_ylabel('Standard deviation (MHz)')
#ax2[0].legend()
#ax2[1].legend()

fig3,ax3 = plt.subplots(2,1,constrained_layout=True)
for j in range(N_exp):
    i = N_exp-j-1
    ax3[0].plot(z_e[idxplot]*1e-3,db(Pb_e[i][idxplot]))
    ax3[1].semilogy(z_e[idxplot]*1e-3,invSNR_pow_e[i][idxplot],label=str(Att[i]),color=colarr[i])
    ax3[1].semilogy(z_e[idx_fit_e]*1e-3,f_model(z_e[idx_fit_e],D1_e[i],D2_e[i],D3_e[i]),
                    label='Model fit',ls='--',color='black')
ax3[0].set_ylim([-40,3])
ax3[1].set_ylim([5e-3,1e0])
ax3[0].set_xlabel('z (km)')
ax3[1].set_xlabel('z (km)')
ax3[0].set_ylabel('Brillouin power (dB)')
ax3[1].set_ylabel('Standard deviation (MHz)')
ax3[1].legend()

fig4,ax4 = plt.subplots(3,2,constrained_layout=True)
ax4b = ax4.flatten()
ax4[0,0].plot(Pp,C1_e)
ax4[0,1].plot(Pp,D1_e)
ax4[0,0].set_ylabel(r'$C_1$ (MHz)')
ax4[0,1].set_ylabel(r'$D_1$')
ax4[0,0].set_ylim([0,np.max(C1_e)*1.1])
ax4[0,1].set_ylim([0,np.max(D1_e)*1.1])
ax4[1,0].plot(Pp,C2_e)
ax4[1,1].plot(Pp,D2_e)
ax4[1,0].set_ylabel(r'$C_2$ (MHz)')
ax4[1,1].set_ylabel(r'$D_2$')
ax4[1,0].set_ylim([0,np.max(C2_e)*1.1])
ax4[1,1].set_ylim([0,np.max(D2_e)*1.1])
ax4[2,0].plot(Pp,alpha_fit_f0_e)
ax4[2,1].plot(Pp,alpha_fit_Pb_e)
ax4[2,0].set_ylabel(r'$\alpha$ ($C_3$)')
ax4[2,1].set_ylabel(r'$\alpha$ ($D_3$)')
ax4[2,0].set_ylim([0,np.max(alpha_fit_f0_e)*1.1])
ax4[2,1].set_ylim([0,np.max(alpha_fit_Pb_e)*1.1])

for axi in ax4b:
    axi.set_xlabel('Pump power (dBm)')


Sase_plot = np.linspace(0,0.8,11)
fig5,ax5 = plt.subplots(1,3,constrained_layout=True)
for i in range(Nsim):
    ax5[0].plot(lamnoise*1e6,Sase_bw[i])
ax5[0].set_xlabel('Wavelength (um)')
ax5[0].set_ylabel('Power spectral density (nW/GHz)')
ax5[1].scatter(Sase_bw_1550,C2_e,label='Meas')
ax5[1].scatter(0,C2,label='Meas (no erbium)')
ax5[1].plot(Sase_plot,C20_e+C21_e*Sase_plot,label='Fit')
ax5[1].set_xlabel(r'$S_{ase}$ at 1550 nm (nW/GHz)')
ax5[1].set_ylabel('C2 (MHz)')
ax5[2].scatter(Sase_bw_1550,D2_e,label='Meas')
ax5[2].scatter(0,D2,label='Meas (no erbium)')
ax5[2].plot(Sase_plot,D20_e+D21_e*Sase_plot,label='Fit')
ax5[2].set_xlabel(r'$S_{ase}$ at 1550 nm (nW/GHz)')
ax5[2].set_ylabel('D2')

# Plot to check whether the oscillations are correlated for different measurements
fig6,ax6 = plt.subplots(constrained_layout=True)
for i in range(N_exp):
    ax6.plot(z_e*1e-3,f0_e[i])
ax6.set_xlabel('z (km)')
ax6.set_ylabel('$f_0$ (MHz)')
ax6.set_ylim([10960,11010])


# %%
