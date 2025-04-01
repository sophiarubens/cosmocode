import numpy as np
from scipy.integrate import quad

Omegam_Planck18=0.3158
OmegaLambda_Planck18=0.6842
H0_Planck18=67.32

def comoving_dist_arg(z,Omegam=Omegam_Planck18,OmegaLambda=OmegaLambda_Planck18):
    return 1/np.sqrt(Omegam*(1+z)**3+OmegaLambda)

def comoving_distance(z,H0=H0_Planck18,Omegam=Omegam_Planck18,OmegaLambda=OmegaLambda_Planck18): # returns value in Mpc
    c=2.998e8
    integral,_=quad(comoving_dist_arg,0,z,args=(Omegam,OmegaLambda,)) #quad(phi_arg_real,0,twopi,args=(phik,phikp,))
    return (c*integral)/(H0*1000)

def z_freq(nu_rest,nu_obs):
    return nu_rest/nu_obs-1

def z_wl(lambda_rest,lambda_obs):
    return lambda_obs/lambda_rest-1