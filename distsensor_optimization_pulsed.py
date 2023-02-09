# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 16:40:10 2022

@author: madshv
"""

import sys
import os
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

# Insert directory of your current folder
sys.path.insert(0, 'C:/Users/madshv/OneDrive - Danmarks Tekniske Universitet/code')

import numpy as np
from scipy.constants import c,h
from numpy import pi,exp,sqrt,log10
from scipy.fft import fft,ifft,fftfreq,fftshift,ifftshift
from scipy.integrate import solve_ivp
from scipy.interpolate import splrep,splev
from scipy.optimize import fsolve
from scipy.signal import peak_widths,find_peaks
import matplotlib.pyplot as plt
from erbium_model.src.fiberdata_erbium import Erbiumfiber_class
from erbium_model.src.simulation_erbium import Erbium_simulation_class
from src.fiberdata_passive import Passivefiber_class
from help_functions import dbm
c = c*1e-9              # Unit m/ns

def Apr0_func(t,T0pr,Ppr0):
    H0 = t>=-T0pr/2
    H1 = t<=T0pr/2
    
    DT = T0pr/30
    H0n = t<=-(T0pr-DT)/2
    H0p = t>=-(T0pr+DT)/2
    H1n = t>=+(T0pr-DT)/2
    H1p = t<=+(T0pr+DT)/2
    H00 = t>=-(T0pr+DT)/2
    H01 = t<=+(T0pr-DT)/2
    
    #return z**0*t**0*sqrt(Ppr0)
    #return sqrt(Ppr0)*(H00*H01+H1n*H1p*(-1/DT*(t-(T0pr+DT)/2))+H0n*H0p*(1/DT*(t+(T0pr-DT)/2)))

    return sqrt(Ppr0)*exp(-(2*t/T0pr)**22)


def gnls(T,W,A0,L,Fiber,Nz,nsaves):
    # Propagates the input pulse A0 using the GNLS split step method.
    # OUTPUT VARIABLES
    # z: List of z-coordinates
    # As: Saved time domain field at distances z
    # INPUT VARIABLES
    # T: Time list
    # W: List of angular frequencies
    # A0: Input pulse
    # L: Propagation length
    # gamma: Nonlinear coefficient
    # beta2: GVD
    # beta3: 3rd order dispersion
    # beta4: 4th order dispersion
    # Nz: No. of steps in progagation (z) direction
    # nsaves: No. of saves of the field along z
    # Fiber: Fiberclass
    
    gr = np.array([Fiber.gr*Fiber.omega[0]/Fiber.omega[1],Fiber.gr])
    gamma = Fiber.gamma
    fr = Fiber.fr
    alpha = Fiber.alpha
    beta2 = Fiber.beta2
    domega = Fiber.omega[0]-Fiber.omega[1]
    dbeta2 = beta2[0]-beta2[1]
    dbeta1 = beta2[1]*domega+dbeta2*domega/2
    
    # Define frequencies
    N = len(T)
            
    # Linear operators D in frequency
    Dp = 1j*(beta2[0]/2*W**2)-alpha[0]/2    # Linear operator
    Dpr = 1j*(dbeta1*W+(beta2[1]/2*W**2))-alpha[1]/2
    
    z = np.linspace(0,L,nsaves)         # List of z-coordinates
    
    # For-loop going through each propagation step
    BF0 = fft(A0,axis=1)             # Initial frequency domain field
    
    Dcon = np.concatenate([Dp,Dpr])
    BFp = BF0[0]
    BFpr = BF0[1]
    
    Ap = np.zeros([N,nsaves],dtype=np.cdouble)
    Apr = np.zeros([N,nsaves],dtype=np.cdouble)
    
    Ap[:,0] = A0[0]
    Apr[:,0] = A0[1]
    for iz in range(1,len(z)):
        BFcon = np.concatenate([BFp,BFpr])
        res = solve_ivp(NonlinOperator,[z[iz-1],z[iz]],BFcon,method='RK45',
                        t_eval=np.array([z[iz]]),rtol=1e-6,atol=1e-8,
                        args=(Dcon,gr,gamma,fr,N))
        BF = res.y
        BFp = BF[0:N,0]
        BFpr = BF[N:,0]
        Ap[:,iz] = ifft(BFp*exp(Dp*z[iz]))
        Apr[:,iz] = ifft(BFpr*exp(Dpr*z[iz]))
        print("Simulation status: %.1f %%" % (iz/(nsaves-1)*100))
    return z,Ap,Apr

def NonlinOperator(z,BF,D,gr,gamma,fr,N):
    Ap = ifft(BF[0:N]*exp(D[0:N]*z))
    Apr = ifft(BF[N:]*exp(D[N:]*z))
    Pp = np.abs(Ap)**2
    Ppr = np.abs(Apr)**2
    NAp =  1j*gamma[0]*(Pp+(2-fr)*Ppr)*Ap-1/2*gr[0]*Ppr*Ap
    NApr = 1j*gamma[1]*(Ppr+(2-fr)*Pp)*Apr+1/2*gr[1]*Pp*Apr
    dBF = np.concatenate([fft(NAp),fft(NApr)])*exp(-D*z)
    return dBF

def prop_EDF(EDFclass,Nz,Pp0,Ppr0,L):
    FORWARD = 1
    BACKWARD = -1
    z_EDF = np.linspace(0,L,Nz)
    no_of_modes = 2     # Number of optical modes in the fiber
    Sim_EDF = Erbium_simulation_class(EDF,no_of_modes,z_EDF)
    Sim_EDF.add_signal(lam_p,Pp0,FORWARD)
    Sim_EDF.add_signal(lam_pr,Ppr0,FORWARD)
    res = Sim_EDF.runan()
    Pp = res[0]
    Ppr = res[1]
    return z_EDF*1e-3,Pp,Ppr

# Physical parameters
Pp0 = 0.1             # CW pump power (W)
Ppr0 = 0.200            # Probe peak power (W)

T0pr = 25               # Probe duration (ns)
lam_p = 1450e-9         # Wavelength (m)
lam_pr = 1550e-9        

f_p = c/lam_p           # Frequency (GHz)
f_pr = c/lam_pr

omega_p = 2*pi*f_p      # Angular frequency rad*GHz
omega_pr = 2*pi*f_pr


# Define fiber parameters
gamma_raman = 1.72e-14#1.72e-14  # Raman gain coefficient (m/W)
df_raman = f_p-f_pr

# Sumitomo Z Fiber LL
alpha_db_p1 = 0.195         # Fiber loss (dB/km)
alpha_db_pr1 = 0.161
D_p1 = 13                   # GVD param (ps/(nm*km))
D_pr1 = 17
Aeff1 = 85e-12
Fiber_SumLL = Passivefiber_class(np.array([lam_p,lam_pr]),\
                           np.array([alpha_db_p1,alpha_db_pr1]),\
                           np.array([D_p1,D_pr1]),\
                           np.array([Aeff1,Aeff1]))
Fiber_SumLL.add_raman(df_raman,gamma_raman/Aeff1)
    
# Sumitomo Z-PLUS Fiber ULL
alpha_db_p1 = 0.180         # Fiber loss (dB/km)
alpha_db_pr1 = 0.153
D_p1 = 16                   # GVD param (ps/(nm*km))
D_pr1 = 20
Aeff1 = 112e-12
Fiber_SumULL = Passivefiber_class(np.array([lam_p,lam_pr]),\
                           np.array([alpha_db_p1,alpha_db_pr1]),\
                           np.array([D_p1,D_pr1]),\
                           np.array([Aeff1,Aeff1]))
Fiber_SumULL.add_raman(df_raman,gamma_raman/Aeff1)
    
# Sumitomo Z-PLUS Fiber 150
alpha_db_p1 = 0.180         # Fiber loss (dB/km)
alpha_db_pr1 = 0.150
D_p1 = 17                   # GVD param (ps/(nm*km))
D_pr1 = 21
Aeff1 = 150e-12
Fiber_Sum150 = Passivefiber_class(np.array([lam_p,lam_pr]),\
                           np.array([alpha_db_p1,alpha_db_pr1]),\
                           np.array([D_p1,D_pr1]),\
                           np.array([Aeff1,Aeff1]))
Fiber_Sum150.add_raman(df_raman,gamma_raman/Aeff1)

dir_edf = r'C:/Users/madshv/OneDrive - Danmarks Tekniske Universitet/fiber_data/ofs_edf/'
file_edf = r'LP980_22841_labversion.s'
#LP980_11841 
EDF = Erbiumfiber_class.from_ofs_files(dir_edf, file_edf)

# Define propagation fibers
L0 = 50e3

L_co = [1]
L_edf = [11.5]
L_fib = [70e3]
C = [1]    # Coupling factor

Fiber_fib0 = Fiber_Sum150
Fiber_pd0 = Fiber_Sum150

Fiber_co = [Fiber_SumULL]
Fiber_fib = [Fiber_SumULL]
Fiber_pd = [Fiber_SumULL]

Nsec = len(L_co)


# Numerical parameters

Tmax = T0pr*10          # Simulation window size (ns)

N = 2**16
dt = Tmax/N
fsam = 1/dt         # Sampling frequency (GHz)
fnyq = fsam/2

Nz = 2001
Nz_save = 101

t = np.linspace(-Tmax/2,Tmax/2,N)
f = fftfreq(N,d=dt)
df = f[1]-f[0]
omega = 2*pi*f
omega_sh = fftshift(omega)
f_sh = omega_sh/(2*pi)


PSDnoise_dbmHz = -121                            # Noise floor (dBm/Hz)
PSDnoise_WHz = 10**(PSDnoise_dbmHz/10)*1e-3     # Noise floor (W/Hz)
PSDnoise_WGHz = PSDnoise_WHz*1e9
ESDnoise_JGHz = 1/2*h*omega_pr*Tmax#PSDnoise_WGHz*Tmax              # Energy spectral density of the noise (J/Hz)

theta_noise_p = pi*np.random.uniform(size=N)
theta_noise_pr = pi*np.random.uniform(size=N)

ASDnoise_p = np.sqrt(ESDnoise_JGHz)*exp(1j*theta_noise_p)    # Amplitude spectral density (sqrt(J/Hz))
ASDnoise_pr = np.sqrt(ESDnoise_JGHz)*exp(1j*theta_noise_pr)

Anoise_p = fft(ASDnoise_p)/Tmax
Anoise_pr = fft(ASDnoise_pr)/Tmax

Ap0 = sqrt(Pp0)*np.ones(N)#+Anoise_p
Apr0 = Apr0_func(t,T0pr,Ppr0)#+Anoise_pr


# %% Fiber propagation


# Section 0 fiber (lead fiber before amplification)
A_init = np.array([Ap0,Apr0])
z_fib,Ap_fib,Apr_fib = gnls(t,omega,A_init,L0,Fiber_fib0,Nz,Nz_save)
Pp_pd_fib = Pp0*exp(-Fiber_pd0.alpha[0]*L0)
Ltot = L0

z = z_fib
Apr = Apr_fib
Ap = Ap_fib

for i in range(0,Nsec):
    # Section of co-propagation
    Ap_start = 0*Ap_fib[:,-1]+sqrt(C[i]*Pp_pd_fib)*np.ones(N)
    Apr_start = Apr_fib[:,-1]
    A_init = np.array([Ap_start,Apr_start])
    z_co,Ap_co,Apr_co = gnls(t,omega,A_init,L_co[i],Fiber_co[i],Nz,Nz_save)
    
    # Section of erbium
    Pp_cw_edf = np.max(np.abs(Ap_co[:,-1])**2)
    Ppr_cw_edf = np.max(np.abs(Apr_co[:,-1])**2)
    
    z_edf,Pp_edf,Ppr_edf = prop_EDF(EDF,Nz_save,Pp_cw_edf,Ppr_cw_edf,L_edf[i])
    Gpr_edf = Ppr_edf/Ppr_edf[0]
    Gp_edf = Pp_edf/Pp_edf[0]
    Ap_edf = Ap_co[:,-1]*sqrt(Gp_edf[-1])
    Apr_edf = Apr_co[:,-1]*sqrt(Gpr_edf[-1])
    
    # Section of fiber
    A_init = np.array([Ap_edf,Apr_edf])
    z_fib,Ap_fib,Apr_fib = gnls(t,omega,A_init,L_fib[i],Fiber_fib[i],Nz,Nz_save)
    
    # Section of parallel pump delivery fiber
    Lsec = L_co[i]+L_edf[i]+L_fib[i]
    Ltot = Ltot+Lsec
    Pp_pd_fib = Pp_pd_fib*(1-C[i])*exp(-Fiber_pd[i].alpha[0]*Lsec)
    
    z = np.concatenate([z,z[-1]+z_co])
    z = np.concatenate([z,z[-1]+z_fib])
    
    Ap = np.concatenate([Ap,Ap_co,Ap_fib],axis=1)
    Apr = np.concatenate([Apr,Apr_co,Apr_fib],axis=1)


# %% Post processing

AFp = ifftshift(ifft(Ap,axis=0),axes=0)*Tmax        # Normalized such that |AF|^2=ESD and sum(|AF|^2)*df=sum(|A|^2)*dt
AFpr = ifftshift(ifft(Apr,axis=0),axes=0)*Tmax

# Brillouin spectrum
dnu = 38e-3     # Spectral FWHM width (GHz)

gb_th = (dnu/2)**2/(f_sh**2+(dnu/2)**2)

Pb_th = np.zeros(AFpr.shape)
PFpr = np.abs(AFpr)**2

for i in range(Pb_th.shape[1]):
    Pb_th[1:,i] = np.convolve(gb_th[1:], PFpr[1:,i],mode='same')


# Find peak frequency

fwidth_peak = np.zeros(Pb_th.shape[1])
f_peak = np.zeros(Pb_th.shape[1])
for i in range(Pb_th.shape[1]):
    Pb_th_spl = splrep(f_sh,Pb_th[:,i])
    f_peak[i] = fsolve(lambda x:splev(x,Pb_th_spl,der=1),0)
    peaks, _ = find_peaks(Pb_th[:,i])
    fwidth_peak[i] = peak_widths(Pb_th[:,i],peaks,rel_height=0.5)[0][0]*df

# %% Save data
issave = 0

filedir = r'C:\Users\madshv\OneDrive - Danmarks Tekniske Universitet\code\longdistance_optimization\singleedfa_100km'
filename = r'T0pr50ns.npz'

if issave==1:
    np.savez_compressed(filedir+'/'+filename,z=z,t=t,f=f,Apr=Apr,Ap=Ap,\
                        Tmax=Tmax,N=N,fsam=fsam,gr=gr,gamma=gamma,fr=fr,\
                        alpha=alpha,beta2=beta2,dbeta1=dbeta1,D=D,\
                        Gain_edfa_p=Gp_edf,Gain_edfa_pr=Gpr_edf,\
                        Aeff=Aeff)


# %% Plotting

PSDpr = np.abs(AFpr)**2

plt.close('all')

idxtstep = 2**1
Tmaxplot = 7*T0pr
Ntplot = int(Tmaxplot/dt)
Z,T = np.meshgrid(z,t[int((N-Ntplot)/2):int((N+Ntplot)/2):idxtstep])

Ppr_plot = np.abs(Apr[int((N-Ntplot)/2):int((N+Ntplot)/2):idxtstep])**2
Pp_plot = np.abs(Ap[int((N-Ntplot)/2):int((N+Ntplot)/2):idxtstep])**2
fig0,ax0=plt.subplots(1,2,constrained_layout=True)
im0a = ax0[0].pcolormesh(T,Z*1e-3,Pp_plot*1e3)
ax0[0].set_xlabel('Time (ns)')
ax0[0].set_ylabel('Distance (m)')
im0b = ax0[1].pcolormesh(T,Z*1e-3,Ppr_plot*1e3)
ax0[1].set_xlabel('Time (ns)')
ax0[1].set_ylabel('Distance (m)')
#cbar0a = plt.colorbar(im0a)
cbar0b = plt.colorbar(im0b)
cbar0b.set_label('Power (mW)')
ax0[0].set_title('Pump')
ax0[1].set_title('Probe')

idxfstep = 2**0
Fmaxplot = fsam/20 #5000/T0pr
Nfplot = int(Fmaxplot/df)
Z,F = np.meshgrid(z,f_sh[int((N-Nfplot)/2):int((N+Nfplot)/2):idxfstep])

AFpr_plot = dbm(np.abs(AFpr[int((N-Nfplot)/2):int((N+Nfplot)/2):idxfstep])**2)
AFp_plot = dbm(np.abs(AFp[int((N-Nfplot)/2):int((N+Nfplot)/2):idxfstep])**2)
fig1,ax1=plt.subplots(1,2,constrained_layout=True)
ax1[0].pcolormesh(F*1e3,Z*1e-3,AFp_plot)
ax1[0].set_xlabel('Frequency (MHz)')
ax1[0].set_ylabel('Distance (km)')
im1b = ax1[1].pcolormesh(F*1e3,Z*1e-3,AFpr_plot)
ax1[1].set_xlabel('Frequency (MHz)')
ax1[1].set_ylabel('Distance (km)')
cbar1b = plt.colorbar(im1b)
cbar1b.set_label('Power (dB)')
im1b.set_clim([np.max(AFp_plot),np.max(AFp_plot)-60])

fig2,ax2 = plt.subplots(2,2,constrained_layout=True)
ax2[0,0].plot(t,np.abs(Ap[:,0])**2/np.max(np.abs(Ap[:,0])**2),label='z=0')
ax2[0,0].plot(t,np.abs(Ap[:,-1])**2/np.max(np.abs(Ap[:,-1])**2),label='z=zmax')
ax2[0,1].plot(t,np.abs(Apr[:,0])**2/np.max(np.abs(Apr[:,0])**2),label='z=0')
ax2[0,1].plot(t,np.abs(Apr[:,-1])**2/np.max(np.abs(Apr[:,-1])**2),label='z=zmax')
ax2[1,0].plot(f_sh,np.abs(AFp[:,0])**2/np.max(np.abs(AFp[:,0])**2),label='z=0')
ax2[1,0].plot(f_sh,np.abs(AFp[:,-1])**2/np.max(np.abs(AFp[:,-1])**2),label='z=zmax')
ax2[1,1].plot(f_sh,np.abs(AFpr[:,0])**2/np.max(np.abs(AFpr[:,0])**2),label='z=0')
ax2[1,1].plot(f_sh,np.abs(AFpr[:,-1])**2/np.max(np.abs(AFpr[:,-1])**2),label='z=zmax')
ax2[1,0].set_xlim([-0.05,0.05])
ax2[1,1].set_xlim([-0.2,0.2])
ax2[0,0].set_xlabel('Time (ns)')
ax2[0,1].set_xlabel('Time (ns)')
ax2[1,0].set_xlabel('Frequency (GHz)')
ax2[1,1].set_xlabel('Frequency (GHz)')
ax2[0,0].set_ylabel('Power (a.u.)')
ax2[0,1].set_ylabel('Power (a.u.)')
ax2[1,0].set_ylabel('Power (a.u.)')
ax2[1,1].set_ylabel('Power (a.u.)')
ax2[0,0].legend()
ax2[0,1].legend()
ax2[1,0].legend()
ax2[1,1].legend()

fig3,ax3 = plt.subplots(constrained_layout=True)
#ax3.plot(z*1e-3,dbm(np.max(np.abs(Ap)**2,axis=0)))
#ax3.plot(z*1e-3,dbm(Pp0*exp(-alpha_p*z)),ls='--')
ax3.plot(z*1e-3,dbm(np.max(np.abs(Apr)**2,axis=0)),label='With pump')
#ax3.plot(z*1e-3,dbm(Ppr0*exp(-alpha_pr*z)),ls='--')
ax3.set_xlabel('z (m)')
ax3.set_ylabel('Power (dBm)')
ax3.legend()


fig4,ax4 = plt.subplots(1,2,constrained_layout=True)
ax4[0].plot(f_sh*1e3,np.abs(AFpr[:,104])**2/np.max(np.abs(AFpr[:,104])**2),label='23.6 km')
ax4[0].plot(f_sh*1e3,np.abs(AFpr[:,142])**2/np.max(np.abs(AFpr[:,142])**2),label='69.2 km')
ax4[0].plot(f_sh*1e3,np.abs(AFpr[:,-1])**2/np.max(np.abs(AFpr[:,-1])**2),label='300.0 km')
ax4[0].plot(f_sh*1e3,gb_th,label=r'$g(f)$',color='black',ls='--')
ax4[0].set_xlim([-100,100])
ax4[0].set_xlabel('Frequency (MHz)')
ax4[0].set_ylabel('Power (Normalized)')
ax4[0].legend()
ax4[0].set_title(r'$P_{pr}(f)$')
ax4[1].plot(f_sh*1e3,Pb_th[:,104]/np.max(Pb_th[:,104]),label='23.6 km')
ax4[1].plot(f_sh*1e3,Pb_th[:,142]/np.max(Pb_th[:,142]),label='69.2 km')
ax4[1].plot(f_sh*1e3,Pb_th[:,-1]/np.max(Pb_th[:,-1]),label='300.0 km')
ax4[1].set_xlim([-80,120])
ax4[1].set_xlabel('Frequency (MHz)')
ax4[1].set_ylabel('Power (Normalized)')
ax4[1].legend()
ax4[1].set_title(r'$g(f)*P_{pr}(f)$')

fig5,ax5 = plt.subplots(1,2,constrained_layout=True)
ax5[0].plot(z*1e-3,f_peak*1e3)
ax5[1].plot(z*1e-3,fwidth_peak*1e3)
ax5[0].set_xlabel('z (m)')
ax5[0].set_ylabel('Peak frequency (MHz)')
ax5[1].set_xlabel('z (m)')
ax5[1].set_ylabel('Brillouin spectrum width (MHz)')

fig5,ax5 = plt.subplots(constrained_layout=True)
ax5b = ax5.twinx()
ax5.plot(z*1e-3,fwidth_peak*1e3,color='tab:blue')
ax5b.plot(z*1e-3,np.max(np.abs(Apr)**2*1e3,axis=0),label='With pump',color='tab:red')
ax5.set_xlabel('z (m)')
ax5.set_ylabel('FWHM width (MHz)',color='tab:blue')
ax5b.set_ylabel('Power (mW)',color='tab:red')
ax5.set_ylim([35,55])
ax5b.set_ylim([0,230])
ax5.legend()

fig6,ax6 = plt.subplots(constrained_layout=True)
idx_tnorm = np.argmin(np.abs(t-185))
ax6.plot(t,np.abs(Apr[:,-1])**2/np.abs(Apr[idx_tnorm,-1])**2)
ax6.set_xlabel('Time (ns)')
ax6.set_ylabel('Normalized power')

