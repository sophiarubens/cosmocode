import numpy as np
from matplotlib import pyplot as plt
from cosmo_distances import *
from power_class import *
from bias_class import *

############################## cosmo params, constants, and conversion factors ########################################################################################################################
Omegam_Planck18=0.3158
Omegabh2_Planck18=0.022383
Omegach2_Planck18=0.12011
OmegaLambda_Planck18=0.6842
lntentenAS_Planck18=3.0448
tentenAS_Planck18=np.exp(lntentenAS_Planck18)
AS_Planck18=tentenAS_Planck18/10**10
ns_Planck18=0.96605
H0_Planck18=67.32
pi=np.pi
nu_rest_21=1420.405751768 # MHz
tiny=1e-100

############################## bundling and preparing Planck18 cosmo params of interest here ########################################################################################################################
scale=1e-9
pars_Planck18=np.asarray([ H0_Planck18, Omegabh2_Planck18,  Omegach2_Planck18,  AS_Planck18,           ns_Planck18])
parnames=                ['H_0',       'Omega_b h**2',      'Omega_c h**2',      '10**9 * A_S',        'n_s'       ]
# parnames_LaTeX=          ['$H_0$',     '$\Omega_b h^2$',   '$\Omega_c h^2$',   '$10**9 * A_S$',      '$n_s$'     ]
pars_Planck18[3]/=scale
nprm=len(pars_Planck18)
dpar=1e-3*np.ones(nprm) # gets overwritten by the adaptive stepper in my numerical differentiator if ill-suited to any case I care to test (although it seems to do okay for my tests so far!)
dpar[3]*=scale

############################## details of a hypothetical survey cooked up for testing purposes ########################################################################################################################
nu_ctr=900. # centre frequency of survey in MHz
channel_width=0.183 # 183 kHz from CHORD Wiki -> SWGs -> Galaxies -> CHORD Pathfinder specs -> Spectral resolution

############################## initializations related to cylindrically binned k-modes ########################################################################################################################
# N_CHORDbaselines=1010 # CHORD-512 (as long as receiver hut gaps remove redundancy only and not unique baselines, as I'm sure they planned)
N_CHORDbaselines=123 # CHORD-64 (same caveat about the gaps)
b_NS_CHORD=8.5 # m
N_NS_CHORD=24
b_EW_CHORD=6.3 # m
N_EW_CHORD=22
bminCHORD=6.3
# bmaxCHORD=np.sqrt((b_NS_CHORD*N_NS_CHORD)**2+(b_EW_CHORD*N_EW_CHORD)**2) # optimistic (includes low-redundancy baselines for CHORD-512) 
bmaxCHORD=np.sqrt((b_NS_CHORD*10)**2+(b_EW_CHORD*7)**2) # pathfinder (as per the CHORD-all telecon on May 26th, but without holes)
kperp_surv=kperp(nu_ctr,N_CHORDbaselines,bminCHORD,bmaxCHORD) # kperp(nu_ctr,N_modes,bmin,bmax)
ceil=275

n_sph_an=1000
n_sph_nu=250

siglos900=4.1 # precalculated for a 900 MHz survey: 0.25*(Dc_ctr-Dc_lo)/10, dialing in the bound set by condition following from linearization...
# hpbw=(1./12.)*pi/180. # what I had been using before the whole synthesized beam vs primary beam mental confusion breaking
hpbw=1./18. # rad; lambda/D estimate (actually physically realistic)

##############################
bundled_gaussian_primary_args=[sig_LoS,beam_fwhm0,beam_fwhm1]
bundled_gaussian_primary_uncs=[epsLoS_test,epsbeam0_test,epsbeam1_test]

nu_window=window_calcs(bminCHORD,bmaxCHORD,
                       ceil,
                       "Gaussian",nu_bundled_gaussian_primary_args,nu_bundled_gaussian_primary_uncs,
                       pars_Planck18,pars_Planck18,
                       n_sph_nu,dpar,
                       nu_ctr,channel_width,
                       pars_forecast_names=parnames)
nu_window.print_survey_characteristics()
nu_window.bias()
nu_window.print_results()
nu_Pcont=nu_window.Pcont_cyl_surv