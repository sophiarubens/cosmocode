import numpy as np
from scipy.integrate import quad

Omegam_Planck18=0.3158
OmegaLambda_Planck18=0.6842
H0_Planck18=67.32
pi=np.pi
twopi=2.*pi
c=2.998e8
nu21=1420.405751768 # MHz
pc=30856775814914000 # m
Mpc=pc*1e6

def comoving_dist_arg(z,Omegam=Omegam_Planck18,OmegaLambda=OmegaLambda_Planck18): # this is 1/ E(z)
    return 1/np.sqrt(Omegam*(1+z)**3+OmegaLambda)

def comoving_distance(z,H0=H0_Planck18,Omegam=Omegam_Planck18,OmegaLambda=OmegaLambda_Planck18): # returns value in Mpc
    integral,_=quad(comoving_dist_arg,0,z,args=(Omegam,OmegaLambda,))
    return (c*integral)/(H0*1000)

def freq2z(nu_rest,nu_obs):
    return nu_rest/nu_obs-1

def z2freq(nu_rest,z):
    return nu_rest/(z+1)

def wl2z(lambda_rest,lambda_obs):
    return lambda_obs/lambda_rest-1

def z2wl(lambda_rest,z):
    return lambda_rest*(z+1)

def kpar(nu_ctr,chan_width,N_chan,H0=H0_Planck18):
    prefac=1e3*twopi*H0*nu21/c # 1e3 to account for units of H0/c ... assumes nu21 and chan_width have the same units
    z_ctr=freq2z(nu21,nu_ctr)
    Ez=1/comoving_dist_arg(z_ctr)
    zterm=Ez/((1+z_ctr)**2*chan_width)
    kparmax=prefac*zterm
    kparmin=kparmax/N_chan
    return np.linspace(kparmin,kparmax,N_chan) # evaluating at the z of the central freq of the survey (trusting slow variation...)

def kperp(nu_ctr,N_baselines,bmin,bmax):
    Dc=comoving_distance(freq2z(nu21,nu_ctr)) # evaluating at the z of the central freq of the survey (trusting slow variation, once again)
    prefac=twopi*nu21*1e6/(c*Dc)
    kperpmin=prefac*bmin
    kperpmax=prefac*bmax
    return np.linspace(kperpmin,kperpmax,N_baselines) # in Mpc^{-1} as long as I use nu21 in MHz, c in m s^{-1}, and Dc in Mpc^{-1}

def wedge_kpar(nu_ctr,kperp,H0=H0_Planck18,nu_rest=nu21):
    z=freq2z(nu_rest,nu_ctr)
    E=1/comoving_dist_arg(z)
    Dc=comoving_distance(z)
    prefac=(H0*Dc*E)/(c*(1+z))
    return prefac*kperp*1e3 # factor of 1e3 to reconcile the m-km mismatch (c in m/s but H0 in km/s/Mpc)