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
beam_fwhm1=(1./12.)*pi/180.
# beam_fwhm1=(1./8.)*pi/180.

############################## initializations related to cylindrically binned k-modes ########################################################################################################################
kpar_surv=kpar(nu_ctr,channel_width,int(N_CHORDcosmo))
ceil=275 # set ceil=0 to recover the untruncated case without having to change any calls
kpar_surv=kpar_surv[:-ceil] # CONSERVATIVE VERSION,, NOT RENAMING TO MINIMIZE THE NUMBER OF REQUIRED CALL UPDATES

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

n_sph_pts_test=450 # this choice is not (yet) mathematically motivated; it's just a reasonable-seeming initial compromise between pedantry and the reality that, at the end of the day, this is an exercise in interpolation
kpar_surv_grid,kperp_surv_grid,Pcyl=unbin_to_Pcyl(kpar_surv,kperp_surv,z_ctr,n_sph_modes=n_sph_pts_test)

fractional_2d_sense=0.1 # Adrian's recommendation: flat 10% uncertainty everywhere as a placeholder
sigma_kpar_kperp=fractional_2d_sense*Pcyl

############################## misc. other initializations for the pipeline functions ########################################################################################################################
epsLoS_test= 0.1 # the epsilons are fractional uncertainties in each epsilon
epsbeam0_test=0.1
epsbeam1_test=0.1
# epsLoS_test=0. # limit works on paper...and in my code as of 15:25 28.05.25 (!!REMEMBER TO RE-CHECK AFTER ANY CODE PARADIGM SHIFTS)
# epsbeam_test=0.

############################## one-stop shop for printed verification of survey characteristics ########################################################################################################################
verbose_test_prints=True
if verbose_test_prints: # fans of well-formatted print statements look away now...
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
    P_cyl_partials=build_cyl_partials(pars_Planck18,z_ctr,n_sph_pts_test,kpar_surv,kperp_surv,dpar)
    np.save("P_cyl_partials.npy",P_cyl_partials)
else:
    P_cyl_partials=np.load("P_cyl_partials.npy")

print("cyl sym case:")
b_cyl_sym_resp=bias( P_cyl_partials,sigma_kpar_kperp,
                     kpar_surv,kperp_surv,
                     sig_LoS,Dc_ctr,beam_fwhm0,
                     pars_Planck18,
                     epsLoS_test,epsbeam0_test,
                    # 0.,0.,
                     z_ctr,n_sph_pts_test,
                     recalc_Pcont=True, 
                     savename="cyl_sym")
printparswbiases(pars_Planck18,parnames,b_cyl_sym_resp )

# assert(1==0), "checking cylindrically symmetric/ analytic sigmas"
print("cyl asym case:")
b_cyl_asym_resp=bias( P_cyl_partials,sigma_kpar_kperp,
                      kpar_surv,kperp_surv,
                      sig_LoS,Dc_ctr,beam_fwhm0,
                      pars_Planck18,
                      epsLoS_test,epsbeam0_test,
                    #   0.,0.,
                      z_ctr,n_sph_pts_test,
                      cyl_sym_resp=False, 
                      fwhmbeam1=beam_fwhm1, epsbeam1=epsbeam1_test,
                    #   fwhmbeam1=beam_fwhm1, epsbeam1=0.,
                      recalc_Pcont=True,
                      savename="cyl_asym")
printparswbiases(pars_Planck18,parnames,b_cyl_asym_resp)

## debug zone to inspect the Pconts more closely for the two cases (this term is responsible for all the differences in the results between the two bias calc strategies at the moment)
Pcont_cyl_sym= np.load("Pcont_cyl_sym.npy")
Pcont_cyl_asym=np.load("Pcont_cyl_asym.npy")
fig,axs=plt.subplots(1,4,figsize=(20,5))
im=axs[0].pcolor(kpar_surv_grid,kperp_surv_grid,Pcont_cyl_sym)

# axs[0].axhline(epsbeam1_test, label="epsbeam1_test")
plt.colorbar(im,ax=axs[0])
axs[0].set_title("Pcont - cyl sym version")
im=axs[1].pcolor(kpar_surv_grid,kperp_surv_grid,Pcont_cyl_asym)
plt.colorbar(im,ax=axs[1])
axs[1].set_title("Pcont - cyl asym version")
im=axs[2].pcolor(kpar_surv_grid,kperp_surv_grid,Pcont_cyl_sym/Pcont_cyl_asym)
plt.colorbar(im,ax=axs[2])
axs[2].set_title("Pcont - ratio of sym/asym versions")
im=axs[3].pcolor(kpar_surv_grid,kperp_surv_grid,Pcont_cyl_sym-Pcont_cyl_asym)
plt.colorbar(im,ax=axs[3])
axs[3].set_title("(Pcont - cyl sym) - (Pcont - cyl asym)")

par_line=1./(np.sqrt(2)*sig_LoS)
perp_line=np.sqrt(ln2)/(Dc_ctr*beam_fwhm1)

for i in range(4):
    axs[i].axhline(perp_line, label="perp sigma for analytical Wtrue", c="C0")
    axs[i].axvline(par_line,  label="par sigma for analytical Wtrue",  c="C1")
    if i==0:
        print("adding par_line and perp_line to plot 0:",par_line,perp_line)

    axs[i].set_xlabel("k$_{||}$ (Mpc$^{-1}$)")
    axs[i].set_ylabel("k$_\perp$ (Mpc$^{-1}$)")

    axs[i].legend()
plt.suptitle("inspection and comparison of Pcont calculation strategies")
plt.tight_layout()
plt.savefig("inspect_compare_Pcont_strats.png")
plt.show()