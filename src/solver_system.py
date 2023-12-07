# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 13:36:27 2022

@author: madshv
"""
# Insert path to the erbium model
import sys
import os
code_path = os.environ.get('CODE_PATH', 
                           'C:/Users/madshv/OneDrive - Danmarks Tekniske Universitet/code')
sys.path.insert(0, code_path)
import numpy as np
from numpy import exp
from scipy.integrate import solve_ivp,simpson
from scipy.constants import h
from scipy.constants import k as kB
from scipy.fft import fft,ifft
from erbium_model.src.simulation_erbium import Erbium_simulation_class


class System_solver_class:
    def __init__(self,Simulation_class):
        S = Simulation_class
        self.Sim = S
        
        self.f_p = S.f_p
        self.f_pr = S.f_pr
        
        self.lam_p = S.lam_p
        self.lam_pr = S.lam_pr
        
        self.Ppr0 = S.Ppr0
        self.Pp0 = S.Pp0
        
        self.Fiber_fib0 = S.Fiber_fib0
        self.Fiber_pd0 = S.Fiber_pd0
        self.Fiber_co = S.Fiber_co
        self.Fiber_edf = S.Fiber_edf
        self.Fiber_fib = S.Fiber_fib
        self.Fiber_pd = S.Fiber_pd
        
        self.L0 = S.L0
        self.L_co = S.L_co
        self.L_edf = S.L_edf
        self.L_fib = S.L_fib
        self.Nsec = S.Nsec
                
        self.C = S.C
        self.Nz = S.Nz
        self.Temp = S.Temp
        
        self.Ppmean0 = 1e-6
        self.no_of_modes = 2
                
        self.lam_noise_init = S.lam_noise_init
        self.idx_noise = S.idx_noise

    def run(self):
        
        #1: Initial transmission fiber section
        z_fib,Pp_fib,Ppr_fib,Snoisefw_fib,Snoisebw_fib = self.prop_transmissionfiber(self.L0,self.Nz,0*self.Ppmean0,self.Ppr0,self.Fiber_fib0)
        Pp_pdf_fib = self.Pp0*exp(-self.Fiber_pd0.alpha[0]*z_fib)

        z = z_fib
        Pp = Pp_fib
        Ppr = Ppr_fib
        Ppmean = self.Ppmean0*exp(-self.Fiber_fib0.alpha[0]*z_fib)
        Pp_pdf = Pp_pdf_fib
        Snoisebw = [Snoisefw_fib]
        Snoisefw = [Snoisebw_fib]
        Gsmall = [Ppr_fib[-1]/Ppr_fib[0]]

        for i in range(0,self.Nsec):
            #2: Co-transmission
            
            Pp_start = Pp_pdf_fib[-1]*self.C[i]+Pp_fib[-1]*(1-self.C[i])
            Ppr_start = Ppr_fib[-1]
            z_co,Pp_co,Ppr_co,Snoisefw_co,Snoisebw_co = self.prop_transmissionfiber(self.L_co[i],self.Nz,Pp_start,Ppr_start,self.Fiber_co[i])
            Ppmean_co = Pp_start*exp(-self.Fiber_co[i].alpha[0]*z_co)
            Pp_pdf_co = Pp_pdf_fib[-1]*(1-self.C[i])*exp(-self.Fiber_pd[i].alpha[0]*z_co)
            Snoisefw.append(Snoisefw_co)
            Snoisebw.append(Snoisebw_co)
            Gsmall.append(Ppr_co[-1]/Ppr_co[0])
            z = np.concatenate([z,z[-1]+z_co])
            
            #3: EDF fiber section
            z_edf,Ppmean_edf,Pprmean_edf,Pnoisefw_edf,Pnoisebw_edf = self.prop_EDF(self.Fiber_edf[i],self.L_edf[i],self.Nz,Ppmean_co[-1],1e-6,self.lam_noise_init)
            Gp_edf = Ppmean_edf/Ppmean_edf[0]
            Gpr_edf = Pprmean_edf/Pprmean_edf[0]
            Pp_edf = Pp_co[-1]*Gp_edf
            Ppr_edf = Ppr_co[-1]*Gpr_edf
            Pp_pdf_edf = Pp_pdf_co[-1]*exp(-self.Fiber_co[i].alpha[0]*z_edf)
            Snoisebw.append(Pnoisebw_edf[self.idx_noise,0])
            Snoisefw.append(Pnoisefw_edf[self.idx_noise,-1])
            Gsmall.append(Gpr_edf[-1])
            z = np.concatenate([z,z[-1]+z_edf])
            
            #4: Transmission fiber section 
            z_fib,Pp_fib,Ppr_fib,Snoisefw_fib,Snoisebw_fib = self.prop_transmissionfiber(self.L_fib[i],self.Nz,Pp_edf[-1],Ppr_edf[-1],self.Fiber_fib[i])
            Ppmean_fib = Ppmean_edf[-1]*exp(-self.Fiber_fib[i].alpha[0]*z_fib)
            Pp_pdf_fib = Pp_pdf_edf[-1]*exp(-self.Fiber_pd[i].alpha[0]*z_fib)
            Snoisefw.append(Snoisefw_fib)
            Snoisebw.append(Snoisebw_fib)
            Gsmall.append(Ppr_fib[-1]/Ppr_fib[0])
            z = np.concatenate([z,z[-1]+z_fib])
            
            Pp = np.concatenate([Pp,Pp_co,Pp_edf,Pp_fib])
            Ppr = np.concatenate([Ppr,Ppr_co,Ppr_edf,Ppr_fib])
            Ppmean = np.concatenate([Ppmean,Ppmean_co,Ppmean_edf,Ppmean_fib])
            Pp_pdf = np.concatenate([Pp_pdf,Pp_pdf_co,Pp_pdf_edf,Pp_pdf_fib])
        Res = System_result_class(z,Pp,Ppr,Ppmean,Pp_pdf,Snoisefw,Snoisebw,Gsmall,self.Sim)
        return Res
            
    def propeq(self,z,P,f_p,f_pr,alpha_p,alpha_pr,gr):
        Pp,Ppr = P
        dPp = -alpha_p*Pp-f_p/f_pr*gr*Pp*Ppr
        dPpr = -alpha_pr*Ppr+gr*Pp*Ppr
        return [dPp,dPpr]
    
    def prop_transmissionfiber(self,L,Nz,Pp0,Ppr0,Fiber):
        zspan = [0,L]
        z = np.linspace(zspan[0],zspan[1],Nz)
        f_p = Fiber.f[0]
        f_pr = Fiber.f[1]
        alpha_p = Fiber.alpha[0]
        alpha_pr = Fiber.alpha[1]
        gr = Fiber.gr
        res = solve_ivp(self.propeq,zspan,[Pp0,Ppr0],method='RK45',
                        t_eval=z,rtol=1e-13,atol=1e-11,args=(f_p,f_pr,alpha_p,alpha_pr,gr))
        Pp = res.y[0]
        Ppr = res.y[1]
        Gfw = Ppr/Ppr[0]
        Gbw = Gfw[-1]/Gfw
        nsp = 1/(1-exp(-h*(self.f_p-self.f_pr)/(kB*self.Temp)))
        Sase_fw = nsp*h*f_pr*gr*Gfw[-1]*simpson(Pp/Gfw,z)
        Sase_bw = nsp*h*f_pr*gr*Gbw[0]*simpson(Pp/Gbw,z)
        return z,Pp,Ppr,Sase_fw,Sase_bw

    def prop_EDF(self,EDFclass,L,Nz,Pp0,Ppr0,lam_noise_init):
        z_EDF = np.linspace(0,L,Nz)
        Sim_EDF = Erbium_simulation_class(EDFclass,z_EDF)
        Sim_EDF.add_fw_signal(self.lam_p,Pp0)
        Sim_EDF.add_fw_signal(self.lam_pr,Ppr0)
        Sim_EDF.add_noise(np.min(lam_noise_init),
                          np.max(lam_noise_init),
                          len(lam_noise_init))
        Res = Sim_EDF.run()
        z_EDF = Res.z
        Pp = Res.Psignal[0]
        Ppr = Res.Psignal[1]
        
        P_noisefw = Res.Pnoise_fw
        P_noisebw = Res.Pnoise_bw
        Nlamnoise = len(P_noisefw)
        S_noisefw = np.zeros(P_noisefw.shape)
        S_noisebw = np.zeros(P_noisebw.shape)
        for i in range(0,len(z_EDF)):
            S_noisefw[:,i] = P_noisefw[:,i]/Sim_EDF.BW_noise[0:Nlamnoise]
            S_noisebw[:,i] = P_noisebw[:,i]/Sim_EDF.BW_noise[0:Nlamnoise]
        return z_EDF,Pp,Ppr,S_noisefw,S_noisebw
    
class System_result_class:
    def __init__(self,z,Pp,Ppr,Ppmean,Pp_pdf,Snoisefw,Snoisebw,Gsmall,Sim_class):
        self.z = z
        self.Pp = Pp
        self.Ppr = Ppr
        self.Ppmean = Ppmean
        self.Pp_pdf = Pp_pdf
        self.Snoisefw = Snoisefw
        self.Snoisebw = Snoisebw
        self.Gsmall = Gsmall
        self.Sim_class = Sim_class
        
        self.Gtotal = np.product(Gsmall)
        
    def calc_SNR(self):
        P_b = self.calc_brillouin_power()
        P_b_end = P_b[-1]*self.Gtotal
        Snoisebw_total = self.calc_noise_power()
        Pb_ref = 0.2*400e-9*self.scatter_coeff()
        C1 = 0.38e6 # Inherent signal fluctuation noise (Hz)
        C2 = 0.19e3*Pb_ref # Shot noise (Hz/W)
        C3 = 1.26e3*Pb_ref # Shot noise (Hz/W)
        SNR = C1+C2/P_b
        return SNR
    
    def scatter_coeff(self):
        S = self.Sim_class
        return S.vg*kB*S.Temp*S.Gamma_b*S.f_pr*S.g_b/(8*S.f_b)*1e18

    def calc_brillouin_power(self):
        S = self.Sim_class
        Epulse = S.Tpulse*self.Ppr*1e-9  # Pulse energy (J)
        return self.scatter_coeff()*Epulse 
        
    def calc_noise_power(self):
        Gsmall = self.Gsmall
        Snoisebw = self.Snoisebw
        Gacc = 1
        Snoisebw_total = Snoisebw[0]*Gacc
        for i in range(len(Gsmall)-1):
            Gacc = Gacc*Gsmall[i]
            Snoisebw_total = Snoisebw_total+Snoisebw[i+1]*Gacc
        return Snoisebw_total


# Code for pulsed propagation
def gnls(T,W,A0,L,Fiber,Nz,nsaves):
    # Propagates the input pulse A0 using the GNLS split step method.
    # OUTPUT VARIABLES
    # z: List of z-coordinates
    # As: Saved time domain field a t distances z
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
    Dp = 1j*(-dbeta1*W+beta2[0]/2*W**2)-alpha[0]/2    # Linear operator
    Dpr = 1j*((beta2[1]/2*W**2))-alpha[1]/2
    
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

def prop_EDF(EDFclass,Nz,Pp0,Ppr0,lam_p,lam_pr,L):
    z_EDF = np.linspace(0,L,Nz)
    no_of_modes = 2     # Number of optical modes in the fiber
    Nlamnoise = 21
    lamnoise_min = 1500*1e-9
    lamnoise_max = 1590*1e-9
    Sim_EDF = Erbium_simulation_class(EDFclass,z_EDF)
    Sim_EDF.add_fw_signal(lam_p,Pp0)
    Sim_EDF.add_fw_signal(lam_pr,Ppr0)
    Sim_EDF.add_noise(lamnoise_min,lamnoise_max,Nlamnoise)
    Res = Sim_EDF.run()
    z_EDF = Res.z
    Pp = Res.Psignal[0]
    Ppr = Res.Psignal[1]
    return z_EDF*1e-3,Pp,Ppr

# Code for pulsed propagation
def gnls1(T,W,A0,L,Fiber,nsaves):
    # Propagates the input pulse A0 using the GNLS split step method.
    # OUTPUT VARIABLES
    # z: List of z-coordinates
    # As: Saved time domain field a t distances z
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
    gamma = Fiber.gamma[1]
    alpha = Fiber.alpha[1]
    beta2 = Fiber.beta2[1]
    
    # Define frequencies
    N = len(T)
        
    # Linear operators D in frequency
    D = 1j*(beta2/2*W**2)-alpha/2
    
    z = np.linspace(0,L,nsaves)         # List of z-coordinates
    
    # For-loop going through each propagation step
    BF = fft(A0)             # Initial frequency domain field
    
    A = np.zeros([N,nsaves],dtype=np.cdouble)
    A[:,0] = A0
    for iz in range(1,len(z)):
        res = solve_ivp(NonlinOperator1,[z[iz-1],z[iz]],BF,method='RK45',
                        t_eval=np.array([z[iz]]),rtol=1e-6,atol=1e-8,
                        args=(D,gamma))
        BF = res.y[:,0]
        A[:,iz] = ifft(BF*exp(D*z[iz]))
        print("Simulation status: %.1f %%" % (iz/(nsaves-1)*100))
    return z,A

def NonlinOperator1(z,BF,D,gamma):
    Apr = ifft(BF*exp(D*z))
    Ppr = np.abs(Apr)**2
    NA = 1j*gamma*Ppr*Apr
    dBF = fft(NA)*exp(-D*z)
    return dBF

