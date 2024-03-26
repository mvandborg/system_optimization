
# %% Load modules


import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

# %% Load data
dir = r'C:\Users\madshv\data_phd\botdr_erbium_data'
file_id_vec = ['17','17_5','18','18_5','19','19_5','20']
dat_vec = []
z_vec = []
spec_vec = []
f_vec = []

for fid in file_id_vec:
    fname = dir+'\data_att'+fid+'.mat'
    dat = loadmat(fname)
    z_vec.append(dat['positions'].flatten())
    f_vec.append(dat['frequencies'].flatten())
    spec_vec.append(dat['spectra'])

N_att = len(file_id_vec)
z = z_vec[0]
f = f_vec[0]
Nz = len(z)
Nf = len(f)

# %% Post processing

idx_noise = (f>10300)*(f<10700)

Pnoise_vec = []
Psig_vec = []

for iatt in range(N_att):
    Pnoise = np.zeros(Nz)
    Psig = np.zeros(Nz)
    for iz in range(Nz):
        Pnoise[iz] = np.std(spec_vec[iatt][idx_noise,iz])
        Psig[iz] = np.max(spec_vec[iatt][:,iz])
    Pnoise_vec.append(Pnoise)
    Psig_vec.append(Psig)

# Fitting of Pnoise(z)=C1*exp(-2*alpha*z)
# log(Pnoise)=log(C1)-2*alpha*z=C+D*z
idx_zfit = (z>35000)*(z<85000)

alpha_fit_vec = []
C_fit_vec = []

for iatt in range(N_att):
    D,C = np.polyfit(z[idx_zfit],np.log(Psig_vec[iatt][idx_zfit]/Pnoise_vec[iatt][idx_zfit]),1)
    C_fit_vec.append(np.exp(C))
    alpha_fit_vec.append(-D/2)
C_fit_vec=np.array(C_fit_vec)
alpha_fit_vec=np.array(alpha_fit_vec)

# %% Simulation
simdir = r"C:\Users\madshv\OneDrive - Danmarks Tekniske Universitet\data_analysis\lios_noise_estimation_data"
data_sim = np.load(simdir+'/200km/simulation_asenoise.npy',allow_pickle=True)
Nsim = len(data_sim[0])

BWnoise = data_sim[6]
BWnoise_lam = data_sim[7]
lamnoise = data_sim[3]
idxlamnoise= np.argmin(lamnoise*1e6<1.55)
Sase_bw = []
Sase_bw_1550 = []
for i in range(Nsim):
    Pnoisebw = data_sim[5][i][:,0]
    Sase_bw_tmp = Pnoisebw/BWnoise_lam*1e-9
    Sase_bw.append( Sase_bw_tmp )
    Sase_bw_1550.append( Sase_bw_tmp[idxlamnoise])
Sase_bw_1550 = np.array(Sase_bw_1550)

# %% plotting
plt.close('all')

iz_vec = np.arange(10000,210000,10000)
Nz_plot = len(iz_vec)

fig0,ax0 = plt.subplots(int(np.ceil(Nz_plot/4)),4,constrained_layout=True)
ax0 = ax0.flatten()
for i in range(Nz_plot):
    iz = iz_vec[i]
    for iatt in range(N_att):
        ax0[i].plot(f_vec[iatt],spec_vec[iatt][:,iz])
    ax0[i].set_title('z = '+str(int(z_vec[iatt][iz]/1e3))+' km')
    ax0[i].set_xlabel('Freq (MHz)')
    
fig1,ax1 = plt.subplots(1,2,constrained_layout=True)
for iatt in range (N_att):
    ax1[0].semilogy(z/1e3,Pnoise_vec[iatt])
    ax1[1].semilogy(z/1e3,Psig_vec[iatt]/Pnoise_vec[iatt])
    ax1[1].semilogy(z/1e3,C_fit_vec[iatt]*np.exp(-2*alpha_fit_vec[iatt]*z),
                    ls='--',color='black')
ax1[0].set_xlabel('z (km)')
ax1[0].set_ylabel('Noise power (a.u.)')
ax1[1].set_xlabel('z (km)')
ax1[1].set_ylabel('SNR')

fig2,ax2 = plt.subplots(constrained_layout=True)
ax2.plot(Sase_bw_1550/Sase_bw_1550[0],label='Simulation')
ax2.plot(C_fit_vec/C_fit_vec[0],label='Experiment')
ax2.set_xlabel('Attenuation')
ax2.set_ylabel('Relative PSD of ASE')
ax2.legend()

# %%
