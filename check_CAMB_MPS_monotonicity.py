import numpy as np
from bias_helper_fcns import get_mps

Omegam_Planck18=0.3158
Omegabh2_Planck18=0.022383
Omegach2_Planck18=0.12011
OmegaLambda_Planck18=0.6842
lntentenAS_Planck18=3.0448
tentenAS_Planck18=np.exp(lntentenAS_Planck18)
AS_Planck18=tentenAS_Planck18/10**10
ns_Planck18=0.96605
H0_Planck18=67.32
infty=np.infty
pi=np.pi
twopi=2.*pi
ln2=np.log(2)
nu_rest_21=1420.405751768 # MHz

h_Planck18=H0_Planck18/100.
Omegamh2_Planck18=Omegam_Planck18*h_Planck18**2
pars_set_cosmo_Planck18=[H0_Planck18,Omegabh2_Planck18,Omegamh2_Planck18,AS_Planck18,ns_Planck18]

test_k,test_P=get_mps(pars_set_cosmo_Planck18,4.5)
print("test k monotonicity:",np.all(np.diff(test_k) > 0))
test_k,test_P=get_mps(pars_set_cosmo_Planck18,13)
print("test k monotonicity:",np.all(np.diff(test_k) > 0))
test_k,test_P=get_mps(pars_set_cosmo_Planck18,2)
print("test k monotonicity:",np.all(np.diff(test_k) > 0))
test_k,test_P=get_mps(pars_set_cosmo_Planck18,0.5)
print("test k monotonicity:",np.all(np.diff(test_k) > 0))