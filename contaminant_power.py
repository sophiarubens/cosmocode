import numpy as np
# from matplotlib.colors import LinearSegmentedColormap
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
# ceil=0 # takes way too long to evaluate
# ceil=200 # apparently also takes way too long to evaluate
# ceil=230 # 485 s for the test to run (in the form it was in around 09:00 on Thursday.)

n_sph_nu=250

hpbw_x= 6*  pi/180. # rad; lambda/D estimate (actually physically realistic)
hpbw_y= 4.5*pi/180.

# epsilons_xy=[0.05]
# epsilons_xy=[0.05,0.3]
epsilons_xy=np.arange(0.0,0.2,0.05)
N_systematic_cases=len(epsilons_xy)
blues_here = plt.cm.Blues( np.linspace(1,0.2,N_systematic_cases))
redo_window_calc=False
suffixes=["systematic-laden and fiducially beamed side-by-side","fractional difference of systematic-laden and fiducially beamed","contaminant beamed, fractional"]
powers=[["Power\n","Dimensionless power\n"],["",""],["",""]]
ylabels=["P (K$^2$ Mpc$^3$)","Δ$^2$ (log(K$^2$/K$^2$)"]
labels=["fiducial","systematic-laden"]

# fig,axs=plt.subplots(1,2,figsize=(12,5))
fig,axs=plt.subplots(3,2,figsize=(12,12))
for i in range(3):
    for j in range(2):
        axs[i,j].set_ylabel(ylabels[j])
        axs[i,j].set_xlabel("k (1/Mpc)")
        axs[i,j].set_title(powers[i][j]+suffixes[i])
for i,epsilon_xy in enumerate(epsilons_xy):
    epsxy=epsilon_xy

    bundled_gaussian_primary_args=[hpbw_x,hpbw_y]
    bundled_gaussian_primary_uncs=[epsxy,epsxy]

    nu_window=window_calcs(bminCHORD,bmaxCHORD,
                            ceil,
                            "AiryGaussian",bundled_gaussian_primary_args,bundled_gaussian_primary_uncs,
                            pars_Planck18,pars_Planck18,
                            n_sph_nu,dpar,
                            nu_ctr,channel_width,
                            pars_forecast_names=parnames, no_monopole=False)
    nu_window.print_survey_characteristics()
    if redo_window_calc:
        t0=time.time()
        nu_window.calc_Pcont_asym()
        t1=time.time()
        print("Pcont calculation time was",t1-t0)

        Pcont_cyl_surv=nu_window.Pcont_cyl_surv
        Ptrue_cyl_surv=nu_window.Ptrue_cyl_surv
        Pthought_cyl_surv=nu_window.Pthought_cyl_surv
        Pfidu_sph=nu_window.Ptruesph # self.ksph,self.Ptruesph
        kfidu_sph=nu_window.ksph

        np.save("Pcont_cyl_surv_"+str(i)+".npy",Pcont_cyl_surv)
        np.save("Pthought_cyl_surv_"+str(i)+".npy",Pthought_cyl_surv)
        np.save("Ptrue_cyl_"+str(i)+".npy",Ptrue_cyl_surv)
        np.save("Pfidu_sph_"+str(i)+".npy",Pfidu_sph)
        np.save("kfidu_sph_"+str(i)+".npy",kfidu_sph)
    else:
        Pcont_cyl_surv=np.load("Pcont_cyl_surv_"+str(i)+".npy")
        Pthought_cyl_surv=np.load("Pthought_cyl_surv_"+str(i)+".npy")
        Ptrue_cyl_surv=np.load("Ptrue_cyl_"+str(i)+".npy")
        Pfidu_sph=np.load("Pfidu_sph_"+str(i)+".npy")
        kfidu_sph=np.load("kfidu_sph_"+str(i)+".npy")

    N_sph=128
    kmin_surv=nu_window.kmin_surv
    kmax_surv=nu_window.kmax_surv
    k_sph=np.linspace(kmin_surv,kmax_surv,N_sph)
    kpar=nu_window.kpar_surv
    kperp=nu_window.kperp_surv
    k00,k11=np.meshgrid(kpar,kperp,indexing="ij")
    kcyl_for_interp=(kpar,kperp)
    Pfidu_sph=np.reshape(Pfidu_sph,(Pfidu_sph.shape[-1],))

    doubled=np.linspace(kmin_surv,kmax_surv,N_sph*2) # print statements offer no reason why this should have to be a workaround
    Pcont_sph=interpn(kcyl_for_interp, Pcont_cyl_surv, doubled, bounds_error=False, fill_value=None)
    Pthought_sph=interpn(kcyl_for_interp, Pthought_cyl_surv, doubled, bounds_error=False, fill_value=None)
    Ptrue_sph=interpn(kcyl_for_interp, Ptrue_cyl_surv, doubled, bounds_error=False, fill_value=None)
    Delta2_fidu=kfidu_sph**3*Pfidu_sph/(2*pi**2)
    Delta2_thought=k_sph**3*Pthought_sph/(2*pi**2)
    Delta2_true=k_sph**3*Ptrue_sph/(2*pi**2)
    Delta2_cont=k_sph**3*Pcont_sph/(2*pi**2)
    P=[Pfidu_sph,Pthought_sph]
    Delta2=[Delta2_fidu,Delta2_thought]
    kmodes=[kfidu_sph,k_sph]
    spectra=[P,Delta2]

    fidu=[Pfidu_sph,Delta2_fidu]
    thought=[Pthought_sph,Delta2_thought]
    true=[Ptrue_sph,Delta2_true]
    cont=[Pcont_sph,Delta2_cont]

    for k,case in enumerate(fidu):
        label_for_eps="frac. unc. in HPBW= "+str(epsxy)
        axs[0,k].loglog(k_sph,true[k],label="fiducially beamed data",c="C1")
        axs[0,k].loglog(k_sph,thought[k],label=label_for_eps,c=blues_here[i])

        axs[1,k].plot(k_sph,(true[k]-thought[k])/true[k],label=label_for_eps,c=blues_here[i])

        axs[2,k].plot(k_sph,cont[k]/true[k],label=label_for_eps,c=blues_here[i])

        for m in range(3):
            axs[m,k].set_xlim(k_sph[1],k_sph[-1])
    axs[0,1].legend()
plt.suptitle("900 MHz CHORD-512 survey (high kperp truncated)\nAiry HPBW {:5.3} (x) and {:5.3} (y)".format(hpbw_x,hpbw_y))
plt.tight_layout()
plt.savefig("contaminant_power_test.png")
plt.show()

# iterate over different epsilons
# plot each curve as a different shade of the same colour map