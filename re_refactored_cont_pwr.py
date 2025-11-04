import numpy as np
# from matplotlib.colors import LinearSegmentedColormap
from matplotlib import pyplot as plt
from scipy.interpolate import interpn
from cosmo_distances import *
from forecasting_pipeline import *
import time

#################################################################################################################################################
################################################ WHAT KIND OF TEST SURVEY ARE YOU INTERESTED IN? ################################################
#################################################################################################################################################
redo_window_calc=True

mode="pathfinder" 
# mode="full" # Nvox way too high to do a practical local run with ceil~0
nu_ctr=363. # centre frequency of survey in MHz
ceil=100
frac_tol_conv=0.1
N_sph=256 # how many spherical modes to put in your theory power spectrum or bin final power spectra down to

# # test 1: UAA, Airy beam
# categ="UAA"
uaa_beam_type="Airy"
N_fid_b_types=1
N_pert_types=1

# # tests 2 and 3: PA, 100 perturbed beams
categ="PA"
N_pert_types=2
PA_N_pbws_pert=100
per_channel_systematic="D3A_like" # None

# test 2: random
# PA_dist="random"
# N_fid_b_types=3
# f_types_prefacs=np.linspace(0.85,1.15,N_fid_b_types) # trivial for now, but it will be less trivial later

# # test 3: corners
PA_dist="corner"
N_fid_b_types=4
f_types_prefacs=np.linspace(0.85,1.15,N_fid_b_types)

# test 4: manual (if I end up needing more flexibility)

#################################################################################################################################################
#################################################################################################################################################
#################################################################################################################################################


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
pars_Planck18[3]/=scale # A_s management (avoid numerical conditioning–related issues)
nprm=len(pars_Planck18)
dpar=1e-3*np.ones(nprm) # starting point (numerical derivatives have adaptive step size)
dpar[3]*=scale

############################## other survey management factors ########################################################################################################################
nu_ctr_Hz=nu_ctr*1e6
wl_ctr_m=c/nu_ctr_Hz
channel_width=0.183 # 183 kHz from CHORD Wiki -> SWGs -> Galaxies -> CHORD Pathfinder specs -> Spectral resolution (there's no more recent estimate, even from telecon slides, as far as I can tell, although a lot of the info on that page is out of date, e.g. f/D ratio reads 0.21 and not 0.25; pathfinder quotes as 11x6 instead of 10x7...)

############################## baselines and beams ########################################################################################################################
b_NS_CHORD=8.5 # m
N_NS_CHORD=24
b_EW_CHORD=6.3 # m
N_EW_CHORD=22
bminCHORD=6.3

if (mode=="pathfinder"): # 10x7=70 antennas (64 w/ receiver hut gaps), 123 baselines
    bmaxCHORD=np.sqrt((b_NS_CHORD*10)**2+(b_EW_CHORD*7)**2) # pathfinder (as per the CHORD-all telecon on May 26th, but without holes)
elif mode=="full": # 24x22=528 antennas (512 w/ receiver hut gaps), 1010 baselines
    bmaxCHORD=np.sqrt((b_NS_CHORD*N_NS_CHORD)**2+(b_EW_CHORD*N_EW_CHORD)**2)

hpbw_x= wl_ctr_m/D *  pi/180. # rad; lambda/D estimate (actually physically realistic)
hpbw_y= 0.75 * hpbw_x         # we know this tends to be a little narrower, based on measurements (...from D3A ...so far)

############################## pipeline administration ########################################################################################################################
epsilons_xy=np.arange(0.0,0.25,0.05) 
N_systematic_cases=len(epsilons_xy)
blues_here = plt.cm.Blues( np.linspace(1,0.2,N_systematic_cases))
oranges_here = plt.cm.Oranges( np.linspace(1,0.2,N_systematic_cases))
ptail="_"+categ+".npy"

plot="P" # "Delta2"
ioname=mode+"_"+str(int(nu_ctr))+"_MHz_"+categ+"_ceil_"+str(ceil)+"_Poisson_"+str(round(frac_tol_conv,1))+"_PA_dist_"+PA_dist+"per_channel_systematic_"+str(per_channel_systematic)

if plot=="P":
    qty_title="Power"
    y_label="P (K$^2$ Mpc$^3$)"
elif plot=="Delta2":
    qty_title="Dimensionless power"
    y_label="Δ$^2$ (log(K$^2$/K$^2$)"

fig,axs=plt.subplots(1,2,figsize=(12,5))
for i in range(2):
    axs[i].set_xlabel("k (1/Mpc)")
    axs[i].set_ylabel(y_label)
axs[0].set_title("side-by-side")
axs[1].set_title("fractional difference")
for i,epsilon_xy in enumerate(epsilons_xy):
    epsxy=epsilon_xy

    if categ!="manual":
        bundled_non_manual_primary_aux=np.array([hpbw_x,hpbw_y])
        bundled_non_manual_primary_uncs=np.array([epsxy,epsxy])
        if categ=="UAA":
            windowed_survey=beam_effects( bminCHORD,bmaxCHORD,ceil,
                                          categ,uaa_beam_type,bundled_non_manual_primary_aux,bundled_non_manual_primary_uncs,
                                          pars_Planck18,pars_Planck18,
                                          N_sph,dpar,
                                          nu_ctr,channel_width,
                                          frac_tol_conv=frac_tol_conv,
                                          pars_forecast_names=parnames, no_monopole=False)
            categ_title="primary beam widths perturbed uniformly across the array"
        elif categ=="PA":
            windowed_survey=beam_effects(bminCHORD,bmaxCHORD,ceil,
                                         categ,"Gaussian",bundled_non_manual_primary_aux,bundled_non_manual_primary_uncs, 
                                         pars_Planck18,pars_Planck18,
                                         N_sph,dpar,
                                         nu_ctr,channel_width,
                                         frac_tol_conv=frac_tol_conv,
                                         pars_forecast_names=parnames, no_monopole=False,
                                         PA_N_pert_types=N_pert_types,PA_N_pbws_pert=PA_N_pbws_pert,PA_pbw_pert_frac=epsxy,
                                         PA_ioname=ioname,PA_recalc=redo_window_calc,PA_distribution=PA_dist,
                                         PA_N_fiducial_beam_types=N_fid_b_types,PA_fidu_types_prefactors=f_types_prefacs) # GET RID OF THE PA_PBW_PERT_FRAC ARGUMENT BC OF OVERLAP WITH PRIMARY_BEAM_UNCS (ONCE THIS QUASI-SPAGHETTI RUNS)
            if PA_dist=="random":
                PA_title=" antennas' primary beam widths perturbed randomly throughout the array"
            elif PA_dist=="corner":
                PA_title="primary beam width categories in separate corners"
            PA_title
            categ_title=str(PA_N_pbws_pert)+PA_title
    else:
        head="placeholder_fname_manual_"
        xy_vec=np.load(head+"_xy_vec.npy")
        z_vec=np.load(head+"_z_vec.npy")
        fidu=np.load(head+"_fidu.npy")
        pert=np.load(pert+"_pert.npy")

        manual_primary_aux=[fidu,pert]
        windowed_survey=beam_effects(bminCHORD,bmaxCHORD,ceil,
                                     ceil,
                                     categ,None,manual_primary_aux,None,
                                     pars_Planck18,pars_Planck18,
                                     N_sph,dpar,
                                     nu_ctr,channel_width,
                                     frac_tol_conv=frac_tol_conv,
                                     pars_forecast_names=parnames, no_monopole=False,
                                     manual_primary_beam_modes=(xy_vec,xy_vec,z_vec))

    windowed_survey.print_survey_characteristics()
    if redo_window_calc:
        t0=time.time()
        windowed_survey.calc_Pcont_asym()
        t1=time.time()
        print("Pcont calculation time was",t1-t0)

        Pcont_cyl_surv=windowed_survey.Pcont_cyl_surv
        Ptrue_cyl_surv=windowed_survey.Ptrue_cyl_surv
        Pthought_cyl_surv=windowed_survey.Pthought_cyl_surv
        Pfidu_sph=windowed_survey.Ptruesph
        kfidu_sph=windowed_survey.ksph

        np.save("Pcont_cyl_surv_"+str(i)+ptail,Pcont_cyl_surv)
        np.save("Pthought_cyl_surv_"+str(i)+ptail,Pthought_cyl_surv)
        np.save("Ptrue_cyl_"+str(i)+ptail,Ptrue_cyl_surv)
        np.save("Pfidu_sph_"+str(i)+ptail,Pfidu_sph)
        np.save("kfidu_sph_"+str(i)+ptail,kfidu_sph)
    else:
        Pcont_cyl_surv=np.load("Pcont_cyl_surv_"+str(i)+ptail)
        Pthought_cyl_surv=np.load("Pthought_cyl_surv_"+str(i)+ptail)
        Ptrue_cyl_surv=np.load("Ptrue_cyl_"+str(i)+ptail)
        Pfidu_sph=np.load("Pfidu_sph_"+str(i)+ptail)
        kfidu_sph=np.load("kfidu_sph_"+str(i)+ptail)

    kmin_surv=windowed_survey.kmin_surv
    kmax_surv=windowed_survey.kmax_surv
    k_sph=np.linspace(kmin_surv,kmax_surv,N_sph)
    kpar=windowed_survey.kpar_surv
    kperp=windowed_survey.kperp_surv
    kcyl_for_interp=(kpar,kperp)
    Pfidu_sph=np.reshape(Pfidu_sph,(Pfidu_sph.shape[-1],))

    doubled=np.linspace(kmin_surv,kmax_surv,N_sph*2) # print statements offer no reason why this should have to be a workaround (I think it was because I had conflicting versions of N_sph that were named confusingly and they were off by almost a factor of two: 128 and 250)
    Pcont_sph=interpn(kcyl_for_interp, Pcont_cyl_surv, doubled, bounds_error=False, fill_value=None)
    Pthought_sph=interpn(kcyl_for_interp, Pthought_cyl_surv, doubled, bounds_error=False, fill_value=None)
    Ptrue_sph=interpn(kcyl_for_interp, Ptrue_cyl_surv, doubled, bounds_error=False, fill_value=None)

    Delta2_fidu=kfidu_sph**3*Pfidu_sph/(2*pi**2)
    Delta2_thought=k_sph**3*Pthought_sph/(2*pi**2)
    Delta2_true=k_sph**3*Ptrue_sph/(2*pi**2)
    Delta2_cont=k_sph**3*Pcont_sph/(2*pi**2)

    fidu=[Pfidu_sph,Delta2_fidu]
    thought=[Pthought_sph,Delta2_thought]
    true=[Ptrue_sph,Delta2_true]
    cont=[Pcont_sph,Delta2_cont]

    if plot=="P":
        k=0
    elif plot=="Delta2":
        k=1

    if i==0:
        fid_label="fiducially beamed data"
        label_for_dot="seeded fractional uncertainty in HPBW"
    else:
        fid_label=""
        label_for_dot=""
    label_for_eps="frac. unc. in HPBW= "+str(np.round(epsxy,2))
    axs[0].semilogy(k_sph,true[k],label=fid_label,c=oranges_here[i])
    axs[0].semilogy(k_sph,thought[k],label=label_for_eps,c=blues_here[i])

    frac_dif=(true[k]-thought[k])/true[k]
    axs[1].plot(k_sph,frac_dif,c=blues_here[i])

    for m in range(2):
        axs[m].set_xlim(kmin_surv,kmax_surv*0.75) # /2 in kmax b/c of +/- in box
frac_dif_lim=1.05*np.max(np.abs(frac_dif[:3*N_sph//4]))
axs[1].set_ylim(-frac_dif_lim,frac_dif_lim)
axs[0].legend()
fig.suptitle("{:5} MHz CHORD {} survey \n" \
             "{}\n" \
             "{} HPBW {:5.3} (x) and {:5.3} (y)\n" \
             "systematic-laden and fiducially beamed {}\n" \
             "{} fiducial beam types; {} beam perturbation types\n" \
             "per-channel systematics status {}"
             "numerical convenience factors: {} high k-parallel channels truncated and Poisson noise averaged to {} pct" \
             "".format(nu_ctr,mode,
                       categ_title,
                       uaa_beam_type,hpbw_x,hpbw_y,
                       qty_title,
                       N_fid_b_types,N_pert_types,
                       per_channel_systematic,
                       ceil, int(frac_tol_conv*100)))
fig.tight_layout()
fig.savefig("Pcont_"+str(plot)+"_"+str(int(nu_ctr))+"_"+str(categ)+"_"+str(N_fid_b_types)+"_fid_b_types_"+str(N_pert_types)+"_pert_types_"+str(ceil)+"_ceil_"+str(round(frac_tol_conv,1))+"_Poisson.png",dpi=200)
fig.show()