import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interpn
from cosmo_distances import *
from forecasting_pipeline import *
import time

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

hpbw_x= 6*  pi/180. # rad; lambda/D estimate (actually physically realistic)
hpbw_y= 4.5*pi/180.

epsLoS=0.1
epsx=0.1
epsy=0.1

##############################
bundled_gaussian_primary_args=[hpbw_x,hpbw_y]
bundled_gaussian_primary_uncs=[epsLoS,epsx,epsy]

redo_window_calc=False
nu_window=window_calcs(bminCHORD,bmaxCHORD,
                        ceil,
                        "AiryGaussian",bundled_gaussian_primary_args,bundled_gaussian_primary_uncs,
                        pars_Planck18,pars_Planck18,
                        n_sph_nu,dpar,
                        nu_ctr,channel_width,
                        pars_forecast_names=parnames)
nu_window.print_survey_characteristics()
if redo_window_calc:
    t0=time.time()
    nu_window.calc_Pcont_asym()
    t1=time.time()
    print("Pcont calculation time was",t1-t0)

    Pcont_cyl=nu_window.Pcont_cyl
    # need to interpolate to survey modes even though I'm not interested in the whole bias calculation... try to short-circuit?
    Pthought_cyl=nu_window.Pthought_cyl
    Pfidu_sph=nu_window.Psph

    np.save("Pcont_cyl.npy",Pcont_cyl)
    np.save("Pthought_cyl.npy",Pthought_cyl)
    np.save("Pfidu_sph.npy",Pfidu_sph)
else:
    Pcont_cyl=np.load("Pcont_cyl.npy")
    Pthought_cyl=np.load("Pthought_cyl.npy")
    Pfidu_sph=np.load("Pfidu_sph.npy")

N_sph=128
k_sph=np.linspace(nu_window.kmin_surv,nu_window.kmax_surv,N_sph)
kpar=nu_window.kpar_surv
kperp=nu_window.kperp_surv
print("kpar.shape,kperp.shape=",kpar.shape,kperp.shape)
k00,k11=np.meshgrid(kpar,kperp,indexing="ij")
# kcyl_for_interp=(k00,k11)
kcyl_for_interp=(kpar,kperp)
Pcont_sph=interpn(kcyl_for_interp, Pcont_cyl, k_sph) #,bounds_error=False, fill_value=None)
Pthought_sph=interpn(kcyl_for_interp, Pthought_cyl, k_sph)
plt.figure()
plt.plot(k_sph,Pfidu_sph,label="fiducial")
plt.plot(k_sph,Pthought_sph,label="systematic-laden") # = from a difference of window functions, NOT a difference of power spectra
plt.xlabel("k (1/Mpc)")
plt.ylabel("power (K$^2$/Mpc$^3$)")
plt.title("contaminant power visualization test")
plt.legend()
plt.savefig("contaminant_power_test.png")
plt.tight_layout()
plt.show()

# ###
# interpolated_slice=interpn(chan_uv_bin_edges,
#                           chan_dirty_image,
#                           (uu_bin_edges_0,vv_bin_edges_0),
#                           bounds_error=False, fill_value=None)
# ###

# re-bin to spherical
# plot
# iterate over different epsilons
# plot each curve as a different shade of the same colour map