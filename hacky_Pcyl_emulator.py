import numpy as np
from matplotlib import pyplot as plt
from bias_helper_fcns import *

Omegam_Planck18=0.3158
Omegabh2_Planck18=0.022383
Omegach2_Planck18=0.12011
OmegaLambda_Planck18=0.6842
lntentenAS_Planck18=3.0448
tentenAS_Planck18=np.exp(lntentenAS_Planck18)
AS_Planck18=tentenAS_Planck18/10**10
ns_Planck18=0.96605
H0_Planck18=67.32
h_Planck18=H0_Planck18/100.
Omegamh2_Planck18=Omegam_Planck18*h_Planck18**2
pars_Planck18=[H0_Planck18,Omegabh2_Planck18,Omegamh2_Planck18,AS_Planck18,ns_Planck18] # suitable for get_mps

def emulate_Pcyl(kpar,kperp,z,pars=pars_Planck18,nsphpts=1000):
    """
    kpar    = k-parallel modes of interest for a cylindrically binned power spectrum emulation (assumed to be monotonic-increasing)
    kperp   = k-perp modes of interest for a cylindrically binned power spectrum emulation (assumed to be monotonic-increasing)
    z       = redshift for which you want the cylindically binned power spectrum
    pars    = cosmo params to use to generate a spherically binned MPS in CAMB
    nsphpts = number of (scalar) k-modes at which the spherically binned CAMB MPS should be sampled
    """
    h=pars[0]/100.
    kmin=np.sqrt(kpar[0]**2+kperp[0]**2)
    kmax=np.sqrt(kpar[-1]**2+kperp[-1]**2)
    k,Psph=get_mps(pars,z,minkh=kmin/h,maxkh=kmax/h,npts=nsphpts) # get_mps(pars,zs,minkh=1e-4,maxkh=1,npts=200)
    Psph=Psph.reshape((Psph.shape[1],))
    kpargrid,kperpgrid=np.meshgrid(kpar,kperp)
    Pcyl=np.zeros((len(kpar),len(kperp)))
    for i,kpar_val in enumerate(kpar):
        for j,kperp_val in enumerate(kperp):
            k_of_interest=np.sqrt(kpar_val**2+kperp_val**2)
            idx_closest_k=np.argmin(np.abs(k-k_of_interest)) # k-scalar in the CAMB MPS closest to the k-magnitude indicated by the kpar-kperp combination for that point in cylindrically binned Fourier space
            Pcyl[i,j]=Psph[idx_closest_k]
    return kpargrid,kperpgrid,Pcyl

test_kpar=np.linspace(0.01,6.10,330) # kpar_surv check: kparmin,kparmax= 0.018657455012629248 6.100987789129764
test_kperp=np.linspace(0.08,3.32,1010) # kperp_surv check: kperpmin,kperpmax= 0.08484655190528016 3.3215296547294053
test_z=0.57
test_kpargrid,test_kperpgrid,test_Pcyl=emulate_Pcyl(test_kpar,test_kperp,test_z)

plt.figure()
plt.imshow(test_Pcyl,extent=[test_kperp[0],test_kperp[-1],test_kpar[-1],test_kpar[0]])
plt.title("check hacky emulator of a cylindrically binned matter power spectrum")
plt.xlabel("k$_{||}$")
plt.ylabel("k$_\perp$")
plt.tight_layout()
plt.show()