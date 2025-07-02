import numpy as np
from matplotlib import pyplot as plt

H0=67.32
h=H0/100.
h2=h**2
Omegam=0.3158
Omegarh2=2.47e-5 # from Ned Caltech (others from Planck DR3 base_plikHM_TTTEEE_lowl_lowE_lensing)
Omegar=Omegarh2/h2
OmegaLambda=0.6842
Omega0=Omegam+Omegar+OmegaLambda
Omega0h2=Omega0*h2
Omegabh2=0.022383
Omegab=Omegabh2/h2
Theta2pt7=2.725
Gamma=Omega0*h
Omegach2=0.12011
Omegac=Omegach2/h2
a1=(46.9*Omega0*h2)**0.670*(1+(32.1*Omega0h2)**(-0.532))
a2=(12.0*Omega0*h2)**0.424*(1+(45.0*Omega0h2)**(-0.582))
alphac=a1**(-Omegab/Omega0)*a2**(-(Omegab/Omega0)**3)
b1=0.944/(1+(458*Omega0h2)**(-0.708))
b2=(0.395*Omega0h2)**(-0.0266)
betacinv=1+b1*((Omegac/Omega0)**b2-1)
betac=1./betacinv
Tc_prefac=alphac*np.log(1.8)*betac/14.2

def calc_transfer(k): # Eisenstein & Hu 1998 eq. 16
    Tb= # baryon sector TF (eq. 13, 21)
    Tc=Tc_prefac* # CDM sector TF (eq. 9, 17)
    return Omegab*Tb/Omega0+Omegac*Tc/Omega0
    # k_agnostic=k/h
    # q=k_agnostic*Theta2pt7**2/Gamma
    # C0=14.2+731/(1+62.5*q)
    # L0=np.log(2*np.e+1.8*q)
    # T0=L0/(L0+C0*q**2)
    # return T0

##

##

k=np.linspace(1e-3,1)
ns=-5

Pfid=P(k)
plt.figure()
plt.loglog(k,Pfid)
plt.xlabel("k")
plt.ylabel("P(k)")
plt.title("primordial power spec power law test")
plt.tight_layout()
plt.show()