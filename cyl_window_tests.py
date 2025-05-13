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
ln2=np.log(2)
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
nu_ctr=900. # centre frequency of survey in MHz
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
channel_width=0.183 # 183 kHz from CHORD Wiki -> SWGs -> Galaxies -> CHORD Pathfinder specs -> Spectral resolution
N_CHORDcosmo=survey_width/channel_width
N_CHORDcosmo_int=int(N_CHORDcosmo)
surv_channels=np.arange(nu_lo,nu_hi,channel_width)
sig_LoS=0.25*(Dc_ctr-Dc_lo)/10 # dialing in the bound set by condition following from linearization...
beam_fwhm=(1./12.)*pi/180. # CHORD pathfinder spec page

############################## initializations related to cylindrically binned k-modes ########################################################################################################################
kpar_surv=kpar(nu_ctr,channel_width,N_CHORDcosmo_int) # kpar(nu_ctr,chan_width,N_chan,H0=H0_Planck18)

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

n_sph_pts_test=450
kpar_surv_grid,kperp_surv_grid,Pcyl=unbin_to_Pcyl(kpar_surv,kperp_surv,z_ctr,nsphpts=n_sph_pts_test)

fractional_2d_sense=0.1 # Adrian's recommendation: flat 10% uncertainty everywhere as a placeholder
sigma_kpar_kperp=fractional_2d_sense*Pcyl

############################## misc. other initializations for the pipeline functions ########################################################################################################################
epsLoS_test= 0.1 # I'm using this as a fractional uncertainty in W calcs...so initialize it like one
epsbeam_test=0.1
# epsLoS_test=0. # the limit works on paper... but does it work in my code? YES! (leave commented as a reminder to check again after any potential future paradigm shifts in my code)
# epsbeam_test=0.

############################## one-stop shop for printed verification of survey characteristics ########################################################################################################################
verbose_test_prints=True
if verbose_test_prints:
    for i in range(3):
        print("........................................................................................")
    print("survey centred at.......................................................................\n    nu ={:>7.4}     MHz \n    z  = {:>9.4} \n    Dc = {:>9.4f}  Mpc\n".format(nu_ctr,z_ctr,Dc_ctr))
    print("survey spans............................................................................\n    nu =  {:>5.4}    -  {:>5.4}    MHz (deltanu = {:>6.4}    MHz) \n    z =  {:>9.4} - {:>9.4}     (deltaz  = {:>9.4}    ) \n    Dc = {:>9.4f} - {:>9.4f} Mpc (deltaDc = {:>9.4f} Mpc)\n".format(nu_lo,nu_hi,survey_width,z_hi,z_lo,deltaz,Dc_hi,Dc_lo,Dc_hi-Dc_lo))
    print("characteristic instrument response widths...............................................\n    sigLoS = {:>7.4}     Mpc (frac. uncert. {:>7.4})\n    beamFWHM = {:>=8.4}  rad (frac. uncert. {:>7.4})\n".format(sig_LoS,epsLoS_test,beam_fwhm,epsbeam_test))
    print("cylindrically binned wavenumbers of the survey..........................................\n    kparallel {:>8.4} - {:>8.4} Mpc^(-1) ({:>4} channels of width {:>7.4}  Mpc^(-1)) \n    kperp     {:>8.4} - {:>8.4} Mpc^(-1) ({:>4} channels of width {:>8.4} Mpc^(-1))\n".format(kpar_surv[0],kpar_surv[-1],len(kpar_surv),kpar_surv[-1]-kpar_surv[-2],   kperp_surv[0],kperp_surv[-1],len(kperp_surv),kperp_surv[-1]-kperp_surv[-2]))
    print("cylindrically binned k-bin sensitivity..................................................\n    fraction of Pcyl amplitude = {:>7.4}".format(fractional_2d_sense))
    for i in range(3):
        print("........................................................................................")

############################## actual pipeline tests ########################################################################################################################
Pcont_cyl=calc_Pcont_cyl(kpar_surv,kperp_surv,
                         sig_LoS,Dc_ctr,beam_fwhm,
                         pars_Planck18,epsLoS_test,epsbeam_test,z_ctr,n_sph_pts_test) 

calc_P_cyl_partials=False
if calc_P_cyl_partials:
    P_cyl_partials=build_cyl_partials(pars_Planck18,z_ctr,n_sph_pts_test,kpar_surv,kperp_surv,dpar) # build_cyl_partials(p,z,nmodes_sph,kpar,kperp,dpar)
    np.save("P_cyl_partials.npy",P_cyl_partials)
else:
    P_cyl_partials=np.load("P_cyl_partials.npy")
b_cyl=bias( P_cyl_partials,sigma_kpar_kperp,
            kpar_surv,kperp_surv,
            sig_LoS,Dc_ctr,beam_fwhm,
            pars_Planck18,
            epsLoS_test,epsbeam_test,
            z_ctr,n_sph_pts_test)

# bias(partials,unc, 
#       kpar,kperp,
#       sigLoS,r0,sigbeam,
#       pars,
#       epsLoS,epsbeam,
#       z,n_sph_pts,
#       beamtype="Gaussian",savestatus=False,savename=None)

printparswbiases(pars_Planck18,parnames,b_cyl)