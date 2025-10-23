import numpy as np
from forecasting_pipeline import *

scale=1e-9
pars_Planck18=np.asarray([ H0_Planck18, Omegabh2_Planck18,  Omegach2_Planck18,  AS_Planck18,           ns_Planck18])
parnames=                ['H_0',       'Omega_b h**2',      'Omega_c h**2',      '10**9 * A_S',        'n_s'       ]
pars_Planck18[3]/=scale
nprm=len(pars_Planck18)
dpar=1e-3*np.ones(nprm) # gets overwritten by the adaptive stepper in my numerical differentiator if ill-suited to any case I care to test (although it seems to do okay for my tests so far!)
dpar[3]*=scale

phead="/Users/sophiarubens/Downloads/research/code/per_antenna/"
ptail="_363_256.npy"
fidu=np.load(phead+"fidu_box"+ptail)
pert=np.load(phead+"pert_box"+ptail)
xy_vec=np.load(phead+"xy_vec_for_boxes"+ptail)
z_vec=np.load(phead+"z_vec_for_boxes"+ptail)
primary_beams=[fidu,pert]
nu_window=window_calcs( 6.5,8,
                        12,
                        "manual",primary_beams,None,
                        pars_Planck18,pars_Planck18,
                        250,dpar,
                        363,0.183,
                        frac_tol_conv=0.1,
                        pars_forecast_names=parnames, no_monopole=False,
                        manual_primary_beam_modes=(xy_vec,xy_vec,z_vec))

# the new part
np.save("kpar_surv.npy",nu_window.kpar_surv)
np.save("kperp_surv.npy",nu_window.kperp_surv)