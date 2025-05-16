import numpy as np
from cosmo_distances import *
from bias_helper_fcns import *

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

############################## bundling and preparing Planck18 cosmo params of interest here ########################################################################################################################
scale=1e-9
pars_Planck18=np.asarray([ H0_Planck18, Omegabh2_Planck18,  Omegach2_Planck18,  AS_Planck18,  ns_Planck18])
parnames=                ['H_0',       'Omega_b h**2',      'Omega_c h**2',      'A_S',        'n_s'       ]
# parnames_LaTeX=          ['$H_0$',     '$\Omega_b h^2$',   '$\Omega_c h^2$',   '$A_S$',      '$n_s$'     ]
pars_Planck18[3]/=scale
nprm=len(pars_Planck18)
dpar=1e-3*np.ones(nprm) # gets overwritten by the adaptive stepper in my numerical differentiator if ill-suited to any case I care to test (although it seems to do okay for my tests so far!)
dpar[3]*=scale

############################## details of a hypothetical survey cooked up for testing purposes ########################################################################################################################
nu_ctr=900. # centre frequency of survey in MHz
channel_width=0.183 # 183 kHz from CHORD Wiki -> SWGs -> Galaxies -> CHORD Pathfinder specs -> Spectral resolution
z_ctr=freq2z(nu_rest_21,nu_ctr)
Dc_ctr=comoving_distance(z_ctr)
survey_width,N_CHORDcosmo=get_channel_config(nu_ctr,channel_width)
nu_lo=nu_ctr-survey_width/2.
z_hi=freq2z(nu_rest_21,nu_lo)
Dc_hi=comoving_distance(z_hi)
nu_hi=nu_ctr+survey_width/2.
z_lo=freq2z(nu_rest_21,nu_hi)
Dc_lo=comoving_distance(z_lo)
deltaz=z_hi-z_lo
surv_channels=np.arange(nu_lo,nu_hi,channel_width)
sig_LoS=0.25*(Dc_ctr-Dc_lo)/10 # dialing in the bound set by condition following from linearization...
beam_fwhm0=(1./12.)*pi/180. # CHORD pathfinder spec page
beam_fwhm1=(1./8.)*pi/180.

############################## initializations related to cylindrically binned k-modes ########################################################################################################################
kpar_surv=kpar(nu_ctr,channel_width,int(N_CHORDcosmo))

N_CHORDbaselines=1010 # upper bound from my formula is 1010 ... might be an overestimate if the gaps in the ultimate grid remove unique baselines instead of just redundancy (although I suspect they were careful to design CHORD so this does not happen)
b_NS_CHORD=8.5 # m
N_NS_CHORD=24
b_EW_CHORD=6.3 # m
N_EW_CHORD=22
bminCHORD=6.3
bmaxCHORD=np.sqrt((b_NS_CHORD*N_NS_CHORD)**2+(b_EW_CHORD*N_EW_CHORD)**2) # fairly optimistic (this baseline only exists 2x in the whole array) # b_NS_CHORD*N_NS_CHORD # b_EW_CHORD*N_EW_CHORD # two options for a case where I'm interested in truncating a bit to avoid looking at such "rare" long baselines
kperp_surv=kperp(nu_ctr,N_CHORDbaselines,bminCHORD,bmaxCHORD) # kperp(nu_ctr,N_modes,bmin,bmax)

n_sph_pts_test=450 # this choice is not (yet) mathematically motivated; it's just a reasonable-seeming initial compromise between pedantry and the reality that, at the end of the day, this is an exercise in interpolation
kpar_surv_grid,kperp_surv_grid,Pcyl=unbin_to_Pcyl(kpar_surv,kperp_surv,z_ctr,n_sph_modes=n_sph_pts_test)

fractional_2d_sense=0.1 # Adrian's recommendation: flat 10% uncertainty everywhere as a placeholder
sigma_kpar_kperp=fractional_2d_sense*Pcyl

############################## misc. other initializations for the pipeline functions ########################################################################################################################
epsLoS_test= 0.1 # the epsilons are fractional uncertainties in each epsilon
epsbeam0_test=0.1
epsbeam1_test=0.1
n_asym_realiz_test=5
# epsLoS_test=0. # the limit works on paper... but does it work in my code? YES! (leave commented as a reminder to check again after any potential future paradigm shifts in my code)
# epsbeam_test=0.

############################## one-stop shop for printed verification of survey characteristics ########################################################################################################################
verbose_test_prints=True
if verbose_test_prints: # fans of well-formated print statements look away now...
    print("survey properties.......................................................................")
    print("........................................................................................")
    print("survey centred at.......................................................................\n    nu ={:>7.4}     MHz \n    z  = {:>9.4} \n    Dc = {:>9.4f}  Mpc\n".format(nu_ctr,z_ctr,Dc_ctr))
    print("survey spans............................................................................\n    nu =  {:>5.4}    -  {:>5.4}    MHz (deltanu = {:>6.4}    MHz) \n    z =  {:>9.4} - {:>9.4}     (deltaz  = {:>9.4}    ) \n    Dc = {:>9.4f} - {:>9.4f} Mpc (deltaDc = {:>9.4f} Mpc)\n".format(nu_lo,nu_hi,survey_width,z_hi,z_lo,deltaz,Dc_hi,Dc_lo,Dc_hi-Dc_lo))
    print("characteristic instrument response widths...............................................\n    sigLoS = {:>7.4}     Mpc (frac. uncert. {:>7.4})\n    beamFWHM = {:>=8.4}  rad (frac. uncert. {:>7.4})\n".format(sig_LoS,epsLoS_test,beam_fwhm0,epsbeam0_test))
    print("specific to the cylindrically asymmetric beam...........................................\n    beamFWHM1 {:>8.4} = rad (frac. uncert. {:>7.4}) \n".format(beam_fwhm1,epsbeam1_test))
    print("cylindrically binned wavenumbers of the survey..........................................\n    kparallel {:>8.4} - {:>8.4} Mpc**(-1) ({:>4} channels of width {:>7.4}  Mpc**(-1)) \n    kperp     {:>8.4} - {:>8.4} Mpc**(-1) ({:>4} channels of width {:>8.4} Mpc**(-1))\n".format(kpar_surv[0],kpar_surv[-1],len(kpar_surv),kpar_surv[-1]-kpar_surv[-2],   kperp_surv[0],kperp_surv[-1],len(kperp_surv),kperp_surv[-1]-kperp_surv[-2]))
    print("cylindrically binned k-bin sensitivity..................................................\n    fraction of Pcyl amplitude = {:>7.4}".format(fractional_2d_sense))

############################## actual pipeline test ########################################################################################################################
calc_P_cyl_partials=False
if calc_P_cyl_partials:
    P_cyl_partials=build_cyl_partials(pars_Planck18,z_ctr,n_sph_pts_test,kpar_surv,kperp_surv,dpar) # build_cyl_partials(p,z,nmodes_sph,kpar,kperp,dpar)
    np.save("P_cyl_partials.npy",P_cyl_partials)
else:
    P_cyl_partials=np.load("P_cyl_partials.npy")
b_cyl_sym_resp=bias( P_cyl_partials,sigma_kpar_kperp,
                     kpar_surv,kperp_surv,
                     sig_LoS,Dc_ctr,beam_fwhm0,
                     pars_Planck18,
                     epsLoS_test,epsbeam0_test,
                     z_ctr,n_sph_pts_test)
printparswbiases(pars_Planck18,parnames,b_cyl_sym_resp )

b_cyl_asym_resp=bias( P_cyl_partials,sigma_kpar_kperp,
                      kpar_surv,kperp_surv,
                      sig_LoS,Dc_ctr,beam_fwhm0,
                      pars_Planck18,
                      epsLoS_test,epsbeam0_test,
                      z_ctr,n_sph_pts_test,
                      cyl_sym_resp=False, 
                      fwhmbeam1=beam_fwhm1, epsbeam1=epsbeam1_test ,n_realiz=n_asym_realiz_test,
                      recalc_Pcont=False) # REMEMBER TO CHANGE BACK TO "TRUE" IF I UPDATE MY PCONT CALCULATION STRATEGY
printparswbiases(pars_Planck18,parnames,b_cyl_asym_resp)

# bias(partials,unc, 
#      kpar,kperp,
#      sigLoS,r0,fwhmbeam0,
#      pars,
#      epsLoS,epsbeam0,
#      z,n_sph_modes,
#      beamtype="Gaussian",save=False,savename=None,
#      cyl_sym_resp=True, 
#      fwhmbeam1=1e-3,epsbeam1=0.1,n_realiz=10,ncubevox=100,
#      recalc_Pcont=False)