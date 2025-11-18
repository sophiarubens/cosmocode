import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import CenteredNorm
from scipy.interpolate import interpn
from cosmo_distances import *
from forecasting_pipeline_local import *
import time

#################################################################################################################################################
################################################ WHAT KIND OF TEST SURVEY ARE YOU INTERESTED IN? ################################################
#################################################################################################################################################
redo_window_calc=True

mode="pathfinder" 
# mode="full" # Nvox way too high to do a practical local run with ceil~0
nu_ctr=363. # centre frequency of survey in MHz
ceil=97
frac_tol_conv=0.04
N_sph=256 # how many spherical modes to put in your theory power spectrum or bin final power spectra down to

# # test 1: UAA, Airy beam
# categ="UAA"
# categ_title="N/A (UNIFORM)"
uaa_beam_type="Airy"
N_fid_b_types=4
N_pert_types=1

# # # tests 2 and 3: PA, 100 perturbed beams
categ="PA"
categ_title=categ
# N_pert_types=2
PA_N_pbws_pert=30
per_channel_systematic= None
# per_channel_systematic="D3A_like"
# per_channel_systematic="sporadic"

# # test 2: random
# PA_dist="random"
# N_fid_b_types=3
f_types_prefacs=np.linspace(0.85,1.15,N_fid_b_types) # trivial for now, but it will be less trivial later

# # # test 3: corners
PA_dist="corner"
# N_fid_b_types=4
# f_types_prefacs=np.linspace(0.85,1.15,N_fid_b_types)

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
# epsilons_xy=np.arange(0.0,0.25,0.05) 
epsilons_xy=[0.02,0.]
N_systematic_cases=len(epsilons_xy)
blues_here = plt.cm.Blues( np.linspace(1,0.2,N_systematic_cases))
oranges_here = plt.cm.Oranges( np.linspace(1,0.2,N_systematic_cases))
ptail="_"+categ+".npy"

plot="P" # "Delta2"
ioname=mode+"_"+str(int(nu_ctr))+"_MHz_"+categ+"_ceil_"+str(ceil)+"_Poisson_"+str(round(frac_tol_conv,2))+"_PA_dist_"+PA_dist+"per_channel_systematic_"+str(per_channel_systematic)

if plot=="P":
    qty_title="Power"
    y_label="P (K$^2$ Mpc$^3$)"
elif plot=="Delta2":
    qty_title="Dimensionless power"
    y_label="Δ$^2$ (log(K$^2$/K$^2$)"

for i,epsilon_xy in enumerate(epsilons_xy):
    fig,axs=plt.subplots(2,2,figsize=(12,10))
    for i in range(2):
        for j in range(2):
            axs[i,j].set_xlabel("k$_{||}$ (1/Mpc)")
            axs[i,j].set_ylabel("k$_{\perp} (1/Mpc)$")
    epsxy=epsilon_xy

    if categ!="manual":
        bundled_non_manual_primary_aux=np.array([hpbw_x,hpbw_y])
        bundled_non_manual_primary_uncs=np.array([epsxy,epsxy])
        if categ=="UAA":
            windowed_survey=beam_effects(
                                            # SCIENCE
                                            # the observation
                                            bminCHORD,bmaxCHORD,                                
                                            nu_ctr,channel_width,                             
                                            evol_restriction_threshold=def_evol_restriction_threshold,    
                                                
                                            # beam generalities
                                            primary_beam_categ=categ,primary_beam_type=uaa_beam_type,       
                                            primary_beam_aux=bundled_non_manual_primary_aux,
                                            primary_beam_uncs=bundled_non_manual_primary_uncs,

                                            # FORECASTING
                                            pars_set_cosmo=pars_Planck18,pars_forecast=pars_Planck18,        
                                            pars_forecast_names=parnames,                           

                                            # NUMERICAL 
                                            n_sph_modes=N_sph,dpar=dpar,                                   
                                            init_and_box_tol=0.05,CAMB_tol=0.05,                           
                                            Nkpar_box=15,Nkperp_box=18,frac_tol_conv=frac_tol_conv,                      
                                            no_monopole=False,                                                    
                                            ftol_deriv=1e-16,maxiter=5,                                      
                                            PA_N_grid_pix=def_PA_N_grid_pix,PA_img_bin_tol=img_bin_tol,        
                                                
                                            # CONVENIENCE
                                            ceil=ceil                                                                                                       
                                            )

            pert_title="primary beam widths perturbed uniformly across the array"
        elif categ=="PA":
            windowed_survey=beam_effects(# SCIENCE
                                         # the observation
                                         bminCHORD,bmaxCHORD,                                                             # extreme baselines of the array
                                         nu_ctr,channel_width,                                                       # for the survey of interest
                                         evol_restriction_threshold=def_evol_restriction_threshold,             # how close to coeval is close enough?
                                             
                                         # beam generalities
                                         primary_beam_categ=categ,primary_beam_type="Gaussian",                 # modelling choices
                                         primary_beam_aux=bundled_non_manual_primary_aux,
                                         primary_beam_uncs=bundled_non_manual_primary_uncs,                          # helper arguments
                                         manual_primary_beam_modes=None,                                        # config space pts at which a pre–discretely sampled primary beam is known

                                         # additional considerations for per-antenna systematics
                                         PA_N_pert_types=N_pert_types,PA_N_pbws_pert=PA_N_pbws_pert,
                                         PA_N_fidu_types=N_fid_b_types,
                                         PA_fidu_types_prefactors=f_types_prefacs,
                                         PA_N_timesteps=def_PA_N_timesteps,PA_ioname=ioname,             # numbers of timesteps to put in rotation synthesis, in/output file name
                                         PA_distribution=PA_dist,mode=mode,
                                         per_channel_systematic=per_channel_systematic,

                                         # FORECASTING
                                         pars_set_cosmo=pars_Planck18,pars_forecast=pars_Planck18,              # implement soon: build out the functionality for pars_forecast to differ nontrivially from pars_set_cosmo
                                         uncs=None,frac_unc=0.1,                                                # for Fisher-type calcs
                                         pars_forecast_names=parnames,                                              # for verbose output

                                         # NUMERICAL 
                                         n_sph_modes=N_sph,dpar=dpar,                                             # conditioning the CAMB/etc. call
                                         init_and_box_tol=0.05,CAMB_tol=0.05,                                   # considerations for k-modes at different steps
                                         Nkpar_box=15,Nkperp_box=18,frac_tol_conv=frac_tol_conv,                          # considerations for cyl binned power spectra from boxes
                                         no_monopole=False,                                                      # enforce zero-mean in realization boxes?
                                         ftol_deriv=1e-16,maxiter=5,                                            # subtract off monopole moment to give zero-mean box?
                                         PA_N_grid_pix=def_PA_N_grid_pix,PA_img_bin_tol=img_bin_tol,            # pixels per side of gridded uv plane, uv binning chunk snapshot tightness
                                            
                                         # CONVENIENCE
                                         ceil=ceil,                                                                # avoid any high kpars to speed eval? (for speedy testing, not science) 
                                         PA_recalc=redo_window_calc                                                        # save time by not repeating per-antenna calculations? 
                                            
                                         )

            if PA_dist=="random":
                PA_title=" randomly throughout the array"
            elif PA_dist=="corner":
                PA_title=" in separate corners"
            PA_title
            pert_title=str(PA_N_pbws_pert)+" primary beam widths perturbed randomly throughout the array"
            categ_title="fiducial beams arranged "+PA_title
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
    kpar_grid,kperp_grid=np.meshgrid(kpar,kperp,indexing="ij")

    frac_dif=Pcont_cyl_surv/Ptrue_cyl_surv
    blues=plt.cm.Blues
    pinkgreen=plt.cm.PiYG
    title_quantities=["P$_{true}$",
                      "P$_{cont}$=P$_{true}$-P$_{thought}$",
                      "P$_{thought}$",
                      "P$_{thought}$/P$_{true}$",
                      "P$_{cont}$/P$_{true}=(P$_{true}$-P$_{thought}$)/P$_{true}$"]
    plot_quantities=[Ptrue_cyl_surv,
                     Pcont_cyl_surv,
                     Pthought_cyl_surv,
                     Pthought_cyl_surv/Ptrue_cyl_surv,
                     frac_dif]
    cmaps=[blues,
           pinkgreen,
           blues,
           pinkgreen,
           pinkgreen]
    vcentres=[None,0,None,1,0]
    for num in range(4):
        i=num//2
        j=num%2
        if (i==2 and j==0):
            j=1
        vcentre=vcentres[num]
        if vcentre is not None:
            norm = CenteredNorm(vcenter=vcentres[num])
        else: 
            norm=None
        im=axs[i,j].pcolor(kpar_grid,kperp_grid,plot_quantities[num],cmap=cmaps[num],norm=norm)
        axs[i,j].set_title(title_quantities[num])
        axs[i,j].set_aspect('equal')
        plt.colorbar(im,ax=axs[i,j],shrink=0.48) # ,shrink=0.75

    # axs[2,0].remove()
    fig.suptitle("{:5} MHz CHORD {} survey \n" \
                "{}\n" \
                "{}\n" \
                "{} HPBW {:5.3} (x) and {:5.3} (y)\n" \
                "systematic-laden and fiducially beamed {}\n" \
                "{} fiducial beam types; {} beam perturbation types\n" \
                "per-channel systematics status {}\n"
                "numerical convenience factors: {} high k-parallel channels truncated and Poisson noise averaged to {} pct" \
                "".format(nu_ctr,mode,
                        pert_title,
                        categ_title,
                        uaa_beam_type,hpbw_x,hpbw_y,
                        qty_title,
                        N_fid_b_types,N_pert_types,
                        per_channel_systematic,
                        ceil, int(frac_tol_conv*100)))
    fig.tight_layout()
    fig.savefig("Pcont_CYL_"+ioname+"EPS_"+str(epsxy)+".png",dpi=200)
    fig.show()