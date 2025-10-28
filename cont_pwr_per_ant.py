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

############################## bundling and preparing Planck18 cosmo params of interest here ########################################################################################################################
scale=1e-9
pars_Planck18=np.asarray([ H0_Planck18, Omegabh2_Planck18,  Omegach2_Planck18,  AS_Planck18,           ns_Planck18])
parnames=                ['H_0',       'Omega_b h**2',      'Omega_c h**2',      '10**9 * A_S',        'n_s'       ]
pars_Planck18[3]/=scale
nprm=len(pars_Planck18)
dpar=1e-3*np.ones(nprm) # gets overwritten by the adaptive stepper in my numerical differentiator if ill-suited to any case I care to test (although it seems to do okay for my tests so far!)
dpar[3]*=scale

############################## details of a hypothetical survey cooked up for testing purposes ########################################################################################################################
nu_ctr=363. # centre frequency of survey in MHz
channel_width=0.183 # 183 kHz from CHORD Wiki -> SWGs -> Galaxies -> CHORD Pathfinder specs -> Spectral resolution

############################## initializations related to cylindrically binned k-modes ########################################################################################################################
# N_CHORDbaselines=1010 # CHORD-512 (as long as receiver hut gaps remove redundancy only and not unique baselines, as I'm sure they planned)
N_CHORDbaselines=123 # CHORD-64 (same caveat about the gaps)
b_NS_CHORD=8.5 # m
N_NS_CHORD=24
b_EW_CHORD=6.3 # m
N_EW_CHORD=22
bminCHORD=6.3
# bmaxCHORD=np.sqrt((b_NS_CHORD*10)**2+(b_EW_CHORD*7)**2) # pathfinder (as per the CHORD-all telecon on May 26th, but without holes)
bmaxCHORD=np.sqrt((b_NS_CHORD*N_NS_CHORD)**2+(b_EW_CHORD*N_EW_CHORD)**2) # the boxes I built are for full CHORD
# ceil=300 # necessary compromise when asking for 0.04 convergence 900 MHz
# ceil=275 # fine for 0.1 Poisson noise level
# ceil=230 # 485 s for the test to run (in the form it was in around 09:00 on Thursday.)
# ceil=12 # fine for the 0.1 Poisson noise level 363 MHz
ceil=0
# frac_tol_conv=0.075
frac_tol_conv=0.1

n_sph_nu=250

hpbw_x= 6*  pi/180. # rad; lambda/D estimate (actually physically realistic)
hpbw_y= 4.5*pi/180.

epsilons_xy=np.arange(0.0,0.3,0.05) # use a smaller vector for a faster test (or just toggle into read-only mode)
# epsilons_xy=np.arange(0.3,0.6,0.05)
N_systematic_cases=len(epsilons_xy)
blues_here = plt.cm.Blues( np.linspace(1,0.2,N_systematic_cases))
oranges_here = plt.cm.Oranges( np.linspace(1,0.2,N_systematic_cases))
redo_window_calc=True
suffixes=["systematic-laden and fiducially beamed side-by-side","fractional difference of systematic-laden and fiducially beamed","contaminant beamed, fractional"]
powers=[["Power\n","Dimensionless power\n"],["",""]]
ylabels=["P (K$^2$ Mpc$^3$)","Δ$^2$ (log(K$^2$/K$^2$)"]
labels=["fiducial","systematic-laden"]

phead="/Users/sophiarubens/Downloads/research/code/per_antenna/"
pmid="_363_256"
ptail="_4_528"
fidu=np.load(phead+"fidu_box"+pmid+".npy")
# pert_box_363_256_4_528_0.0
pert_all=np.zeros((len(epsilons_xy),fidu.shape[0],fidu.shape[1],fidu.shape[2]))
for i,eps in enumerate(epsilons_xy):
    pert_all[i]=np.load(phead+"pert_box"+pmid+ptail+"_"+str(round(eps,2))+".npy")
xy_vec=np.load(phead+"xy_vec_for_boxes"+pmid+".npy")
z_vec=np.load(phead+"z_vec_for_boxes"+pmid+".npy")

print("2pi/(xy_vec[1]-xy_vec[0])= k_{perp,max}=",2*pi/(xy_vec[1]-xy_vec[0]))
print("2pi/(z_vec[1]- z_vec[0])=  k_{perp,max}=",2*pi/(z_vec[1]-z_vec[0]))
# primary_beams=[fidu,pert]

fig,axs=plt.subplots(2,2,figsize=(12,8))
for i in range(2):
    for j in range(2):
        axs[i,j].set_ylabel(ylabels[j])
        axs[i,j].set_xlabel("k (1/Mpc)")
        axs[i,j].set_title(powers[i][j]+suffixes[i])
for i,epsilon_xy in enumerate(epsilons_xy):
    epsxy=epsilon_xy
    primary_beams=[fidu,np.array(pert_all[i])]

    nu_window=window_calcs(bminCHORD,bmaxCHORD,
                            ceil,
                            "manual",primary_beams,None,
                            pars_Planck18,pars_Planck18,
                            n_sph_nu,dpar,
                            nu_ctr,channel_width,
                            frac_tol_conv=frac_tol_conv,
                            pars_forecast_names=parnames, no_monopole=False,
                            manual_primary_beam_modes=(xy_vec,xy_vec,z_vec))
    nu_window.print_survey_characteristics()
    if redo_window_calc:
        t0=time.time()
        nu_window.calc_Pcont_asym()
        t1=time.time()
        print("Pcont calculation time was",t1-t0)

        Pcont_cyl_surv=nu_window.Pcont_cyl_surv
        Ptrue_cyl_surv=nu_window.Ptrue_cyl_surv
        Pthought_cyl_surv=nu_window.Pthought_cyl_surv
        Pfidu_sph=nu_window.Ptruesph
        kfidu_sph=nu_window.ksph

        np.save("Pcont_cyl_surv_"+str(i)+"_per_ant.npy",Pcont_cyl_surv)
        np.save("Pthought_cyl_surv_"+str(i)+"_per_ant.npy",Pthought_cyl_surv)
        np.save("Ptrue_cyl_"+str(i)+"_per_ant.npy",Ptrue_cyl_surv)
        np.save("Pfidu_sph_"+str(i)+"_per_ant.npy",Pfidu_sph)
        np.save("kfidu_sph_"+str(i)+"_per_ant.npy",kfidu_sph)
    else:
        Pcont_cyl_surv=np.load(phead+"Pcont_cyl_surv_"+str(i)+"_per_ant.npy")
        Pthought_cyl_surv=np.load(phead+"Pthought_cyl_surv_"+str(i)+"_per_ant.npy")
        Ptrue_cyl_surv=np.load(phead+"Ptrue_cyl_"+str(i)+"_per_ant.npy")
        Pfidu_sph=np.load(phead+"Pfidu_sph_"+str(i)+"_per_ant.npy")
        kfidu_sph=np.load(phead+"kfidu_sph_"+str(i)+"_per_ant.npy")

    N_sph=256
    kmin_surv=nu_window.kmin_surv
    kmax_surv=nu_window.kmax_surv
    k_sph=np.linspace(kmin_surv,kmax_surv,N_sph)
    kpar=nu_window.kpar_surv
    kperp=nu_window.kperp_surv
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

    fidu=[Pfidu_sph,Delta2_fidu]
    thought=[Pthought_sph,Delta2_thought]
    true=[Ptrue_sph,Delta2_true]
    cont=[Pcont_sph,Delta2_cont]

    for k,case in enumerate(fidu):
        if i==0:
            fid_label="fiducially beamed data"
            label_for_dot="seeded fractional uncertainty in HPBW"
        else:
            fid_label=""
            label_for_dot=""
        label_for_eps="frac. unc. in HPBW= "+str(np.round(epsxy,2))
        axs[0,k].plot(k_sph,true[k],label=fid_label,c=oranges_here[i])
        axs[0,k].plot(k_sph,thought[k],label=label_for_eps,c=blues_here[i])
        axs[0,k].set_ylim(0,1.2*np.max(true[k]))

        axs[1,k].plot(k_sph,(true[k]-thought[k])/true[k],c=blues_here[i])
        axs[1,k].axhline(epsilons_xy[i],c=blues_here[i],ls=":",label=label_for_dot)

        for m in range(2):
            axs[m,k].set_xlim(kmin_surv,kmax_surv/2) # /2 in kmax b/c of +/- in box
axs[0,1].legend()
axs[1,1].legend()
plt.suptitle("900 MHz CHORD-64 survey (high kperp truncated)\nAiry HPBW {:5.3} (x) and {:5.3} (y)\n two perturbation types".format(hpbw_x,hpbw_y))
plt.tight_layout()
plt.savefig("contaminant_power_test_per_ant.png",dpi=200)
# plt.show()







xy_vec_box=np.load(phead+"xy_vec_for_boxes"+ptail)
z_vec_box=np.load(phead+"z_vec_for_boxes"+ptail)

kpar_min_box= twopi/(z_vec_box[-1]-  z_vec_box[0])
kpar_max_box= twopi/(z_vec_box[-1]-  z_vec_box[-2])
kpar_vec_box=np.linspace(kpar_min_box,kpar_max_box,len(xy_vec_box))
kperp_min_box=twopi/(xy_vec_box[-1]-xy_vec_box[0])
kperp_max_box=twopi/(xy_vec_box[-1]-xy_vec_box[-2])
kperp_vec_box=np.linspace(kperp_min_box,kperp_max_box,len(z_vec_box))

kx_grid_box,ky_grid_box,kpar_grid_box=np.meshgrid(kperp_vec_box,kperp_vec_box,kpar_vec_box)
kperp_square=np.sqrt(kx_grid_box**2+ky_grid_box**2)
kperp_grid_box=np.sqrt(kx_grid_box**2+ky_grid_box**2)
k_grid_box=np.sqrt(kperp_grid_box**2+kpar_grid_box**2)

all_histograms=np.zeros((len(kpar_vec_box),len(doubled)))
histogram_bins=np.append(doubled,2*doubled[-1]-doubled[-2]) # need to add another bin to prevent vector length issues
plt.figure(figsize=(12,5))
for i,kpar_value in enumerate(kpar_vec_box):
    print("kpar_vec_box.shape=",kpar_vec_box.shape)
    kmags_slice=np.sqrt(kpar_vec_box[i]**2+kperp_square**2) # SHAPE ISSUE: (132,256) (132,)
    histogram_slice,_=np.histogram(kmags_slice,histogram_bins)
    all_histograms[i]=histogram_slice

plt.stackplot(doubled,all_histograms)
np.savetxt("all_histograms.txt",all_histograms)
plt.xlabel("k (1/Mpc)")
plt.ylabel("number of voxels")
plt.title("bin stats of successive kpar slices")
plt.tight_layout()
plt.savefig("voxel_binning_histogram_"+str(N_sph)+".png")
plt.show()