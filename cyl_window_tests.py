import numpy as np
from matplotlib import pyplot as plt
from cosmo_distances import *
from bias_helper_fcns import *

############################## constants ########################################################################################################################
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
nu_rest_21=1420.405751768 # MHz
c=2.998e8 # m s^{-1}
pc=30856775814914000 # m
Mpc=pc*1e6

############################## bundling and preparing Planck18 cosmo params of interest here ########################################################################################################################
scale=1e-9
pars_Planck18=np.asarray([ H0_Planck18, Omegabh2_Planck18,  Omegach2_Planck18,  AS_Planck18,  ns_Planck18])
parnames=                ['H_0',       'Omega_b h^2',      'Omega_c h^2',      'A_S',        'n_s'       ]
parnames_LaTeX=          ['$H_0$',     '$\Omega_b h^2$',   '$\Omega_c h^2$',   '$A_S$',      '$n_s$'     ]
pars_Planck18[3]/=scale
nprm=len(pars_Planck18)
dpar=1e-3*np.ones(nprm)
dpar[3]*=scale

############################## details of a hypothetical survey cooked up for testing purposes ########################################################################################################################
nu_ctr=900 # centre frequency of survey in MHz
z_ctr=freq2z(nu_rest_21,nu_ctr)
Dc_ctr=comoving_distance(z_ctr)
survey_width=60. # survey bandwidth in MHz ... based on the 1/15 deltanu/nu ratio inspired by HERA cosmological surveys
nu_lo=nu_ctr-survey_width/2.
z_hi=freq2z(nu_rest_21,nu_lo)
Dc_hi=comoving_distance(z_hi)
nu_hi=nu_ctr+survey_width/2.
z_lo=freq2z(nu_rest_21,nu_hi)
Dc_lo=comoving_distance(z_lo)
deltaz=z_hi-z_lo
channel_width=0.183 # 183 kHz from CHORD Wiki -> SWGs -> Galaxies -> Spectral resolution
N_CHORDcosmo=survey_width/channel_width
N_CHORDcosmo_int=int(N_CHORDcosmo)
surv_channels=np.arange(nu_lo,nu_hi,channel_width)
print("survey centred at",nu_ctr,"MHz / z=",z_ctr,"/ D_c=",Dc_ctr,"Mpc")
print("survey spans",nu_lo,"-",nu_hi,"MHz (width=",survey_width,"MHz) in",N_CHORDcosmo_int,"channels of width",channel_width,"MHz")
print("or, in redshift space, z=",z_hi,"-",z_lo,"(deltaz=",deltaz,")")
print("or, in comoving distance terms, D_c=",Dc_hi,"-",Dc_lo,"Mpc")
sig_LoS=0.25*(Dc_ctr-Dc_lo)/10 # dialing in the bound set by condition following from linearization...
print("sig_LoS=",sig_LoS,"Mpc")
CHORD_ish_fwhm_surv=(1./12.)*pi/180. # CHORD pathfinder spec page:
# D3A6 beam measurements in Ian's MSc thesis, taken by inspection from the plot and eyeball-averaged over the x- and y-pols 4 deg = 4pi/180 rad = pi/45 rad # approximate, but specific to this hypothetical 900 MHz survey 

############################## cylindrically binned k-mode initialization ########################################################################################################################
kpar_surv=kpar(nu_ctr,channel_width,N_CHORDcosmo_int) # kpar(nu_ctr,chan_width,N_chan,H0=H0_Planck18)
print("kpar_surv check: kparmin,kparmax=",kpar_surv[0],kpar_surv[-1])

N_CHORDbaselines=1010 # upper bound (b/c not sure if the grid gaps will remove redundance or just unique baselines) from my formula is 1010
b_NS_CHORD=8.5 # m
N_NS_CHORD=24
b_EW_CHORD=6.3 # m
N_EW_CHORD=22
bminCHORD=6.3
bmaxCHORD=np.sqrt((b_NS_CHORD*N_NS_CHORD)**2+(b_EW_CHORD*N_EW_CHORD)**2) # too optimistic ... this is a low-redundancy baseline and the numerics will be better if I don't insist upon being so literal
# bmaxCHORD=b_NS_CHORD*N_NS_CHORD # if I want to truncate to avoid looking at such "rare" long baselines
# bmaxCHORD=b_EW_CHORD*N_EW_CHORD
kperp_surv=kperp(nu_ctr,N_CHORDbaselines,bminCHORD,bmaxCHORD) # kperp(nu_ctr,N_modes,bmin,bmax)

kpar_surv_grid,kperp_surv_grid=np.meshgrid(kpar_surv,kperp_surv)
print("kperp_surv check: kperpmin,kperpmax=",kperp_surv[0],kperp_surv[-1])

############################## misc. other initializations for the pipeline functions ########################################################################################################################
btype="Gaussian" # Airy implementation not completely working yet 
savestat=True
saven="validation"
eps_test=0.01 # ITERATE OVER DIFFERENT VALUES ONCE EVERYTHING AT LEAST RUNS
n_sph_pts_test=450

############################## actual pipeline tests ########################################################################################################################
Pcont_cyl=calc_Pcont_cyl(kpar_surv,kperp_surv,
                         sig_LoS,Dc_ctr,CHORD_ish_fwhm_surv,
                         savestat,saven,btype,
                         pars_Planck18,eps_test,z_ctr,n_sph_pts_test) # calc_Pcont_cyl(kpar,kperp,sigLoS,r0,thetaHWHM,savestatus,savename,beamtype,pars,eps,z,n_sph_pts)
# sigma_kpar_kperp=np.load("cyl_sense_thermal_surv.npy") # NEED TO DEBUG THE INFS
sigma_kpar_kperp=np.exp(-(kperp_surv_grid/2)**2)
# plt.figure()
# plt.imshow(sigma_kpar_kperp)
# plt.colorbar()
# plt.show()

P_cyl=unbin_to_Pcyl(kpar_surv,kperp_surv,z_ctr,nsphpts=n_sph_pts_test) # unbin_to_Pcyl(kpar,kperp,z,pars=pars_Planck18,nsphpts=500)
calc_P_cyl_partials=False
if calc_P_cyl_partials:
    P_cyl_partials=build_cyl_partials(pars_Planck18,z_ctr,n_sph_pts_test,kpar_surv,kperp_surv,dpar) # build_cyl_partials(p,z,nmodes_sph,kpar,kperp,dpar)
    np.save("P_cyl_partials.npy",P_cyl_partials)
else:
    P_cyl_partials=np.load("P_cyl_partials.npy")
F_cyl,B_cyl=fisher_and_B_cyl(P_cyl_partials,sigma_kpar_kperp,
                             kpar_surv,kperp_surv,
                             sig_LoS,Dc_ctr,CHORD_ish_fwhm_surv,
                             savestat,saven,btype,
                             pars_Planck18,eps_test,z_ctr,n_sph_pts_test) # fisher_and_B_cyl(partials,unc, kpar,kperp,sigLoS,r0,thetaHWHM,savestatus,savename,beamtype,pars,eps,z,n_sph_pts)
b_cyl=bias(F_cyl,B_cyl)
printparswbiases(pars_Planck18,parnames,b_cyl)