import numpy as np
from matplotlib import pyplot as plt
from cosmo_distances import *
from forecasting_pipeline import *

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

n_sph_nu=250

# hpbw=(1./12.)*pi/180. # what I had been using before the whole synthesized beam vs primary beam mental confusion breaking
hpbw_x= 6*  pi/180. # rad; lambda/D estimate (actually physically realistic)
hpbw_y= 4.5*pi/180.

epsLoS=0.1
epsx=0.1
epsy=0.1

##############################
bundled_gaussian_primary_args=[hpbw_x,hpbw_y]
bundled_gaussian_primary_uncs=[epsLoS,epsx,epsy]

nu_window=window_calcs(bminCHORD,bmaxCHORD,
                       ceil,
                       "AiryGaussian",bundled_gaussian_primary_args,bundled_gaussian_primary_uncs,
                       pars_Planck18,pars_Planck18,
                       n_sph_nu,dpar,
                       nu_ctr,channel_width,
                       pars_forecast_names=parnames)
# nu_window.print_survey_characteristics()
# nu_window.bias()
# nu_window.print_results()

vec=np.linspace(-4,4,50)
xx,yy,zz=np.meshgrid(vec,vec,vec,indexing="ij")
grid=np.sqrt(xx**2+yy**2+zz**2)
manual_primary_beam_fid=np.abs(grid)*np.exp(-grid**2)+3       # ramped Gaussian
manual_primary_beam_mis=np.abs(grid)*np.exp(-(grid+0.5)**2)+3 # differently-ramped Gaussian
manual_primary_beams_bundled=[manual_primary_beam_fid,manual_primary_beam_mis]

man_window=window_calcs(bminCHORD,bmaxCHORD,
                        ceil,
                        "manual",manual_primary_beams_bundled,bundled_gaussian_primary_uncs,
                        pars_Planck18,pars_Planck18,
                        n_sph_nu,dpar,
                        nu_ctr,channel_width,
                        pars_forecast_names=parnames,
                        manual_primary_beam_modes=(vec,vec,vec))
                        # manual_primary_beam_modes=(xx,yy,zz))
# man_window.print_survey_characteristics()
man_window.bias()
man_window.print_results()