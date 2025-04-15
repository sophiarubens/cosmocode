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

def kperplims(nuctr,bmin,bmax,nurest=nu21): # nuctr here is nu0 in Liu & Shaw 2020 eqs 33-40
    z=freq2z(nurest,nuctr)
    Dc=comoving_distance(z) # Mpc
    print("in kperplims, Dc=",Dc,"Mpc")
    Dc_m=Dc*Mpc # m
    print("in kperplims, Dc_m=",Dc_m,"m")
    # Ez=1/comoving_dist_arg(z)
    kperp_prefac=twopi*nuctr*1e6/(c*Dc_m) # m^{-2} # the *1e6 is to make up for passing a frequency in MHz
    print("in kperplims, kperp_prefac=twopi*nuctr/(c*Dc_m)=",kperp_prefac,"m^{-2}")
    kperpmin=kperp_prefac*bmin # bmin and bmax should be in m
    kperpmax=kperp_prefac*bmax
    return kperpmin,kperpmax # in m^{-1}

# def deltakpar(nuctr,deltanu,nurest=nu21): # Thursday, April 9th's strategy (before I realized the extent to which I'd need to be careful about nonlinearities in z-space for a vector of uniformly-spaced frequencies)
#     z=freq2z(nurest,nuctr)
#     Ez=1/comoving_dist_arg(z)
#     nurest_Hz=nurest*1e6 # no longer using MHz
#     return twopi*H0_Planck18*nurest_Hz*Ez*1e3/(c*(1+z)**2*deltanu) # 1e3 is to take care of the km baked into this value of H0 and cancel them with the m in the numerator of c in the units I'm using ... this gives deltakpar in Mpc^{-1}

# def kpar(surv_channels,N): # pass the vector of frequency channels in MHz to be compatible with freq2z and the MHz value of nu21 I'm currently using
#     surv_z=freq2z(nu21,surv_channels)
#     surv_Ez=1/comoving_dist_arg(surv_z)
#     channel_deltanu=surv_channels[1]-surv_channels[0] # kind of hacky because I assume there is a characteristic channel width / that the channels are evenly spaced
#     return twopi*nu21*H0_Planck18*surv_Ez*1e3/(c*(1+surv_z)**2*N*channel_deltanu) # value in Mpc^{-1}

def kpar(nu_ctr,chan_width,N_chan,H0=H0_Planck18):
    prefac=1e3*twopi*H0*nu21/c
    # print("prefac=",prefac)
    z_ctr=freq2z(nu21,nu_ctr)
    Ez=1/comoving_dist_arg(z_ctr)
    # print("Ez=",Ez)
    # print("zterm_denom=",(1+z_ctr)**2*chan_width)
    zterm=Ez/((1+z_ctr)**2*chan_width)
    # print("zterm=",zterm)
    kparmax=prefac*zterm # 1e3 to account for units of H0/c ... assumes nu21 and chan_width have the same units
    kparmin=kparmax/N_chan
    return np.linspace(kparmin,kparmax,N_chan) # evaluating at the z of the central freq of the survey (trusting slow variation...)