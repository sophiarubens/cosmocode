import numpy as np
from matplotlib import pyplot as plt
# from power_class import *
# from bias_helper_fcns import *
from forecasting_pipeline import *
import time

Lsurvey=126 # 63
Nvox=54 # 10 
Nk = 11 # 8
Nk1=13
# mode="lin"
Nrealiz=100

t0=time.time()
colours=plt.cm.Blues(np.linspace(0.2,1,Nrealiz))

sigma02=1e3 # wide   in config space
sigma12=10  # medium in config space
sigma22=0.5 # narrow in config space
beamfwhm_x=3.5 # kind of hacky but enough to stop it from entering the Delta-like case every single time (for the numerics as of 2025.07.07 09:47, the voxel scale comparison value is ~2.42)
beamfwhm_y=np.copy(beamfwhm_x)

# need to add r0 when calling cosmo_stats directly (even if window_calcs handles it directly)
# r0=comoving_distance(self.z_ctr)
r0=2000
bundled0=(beamfwhm_x,beamfwhm_y,r0)
med=0.1
bundled1=(med*beamfwhm_x,med*beamfwhm_y,r0)
nar=0.01
bundled2=(nar*beamfwhm_x,nar*beamfwhm_y,r0)

# idx=-0.9 # DECAYING   power law
# idx=2.3  # INCREASING power law
idx=0 # flat power spec / white noise box
ktest=np.linspace(twopi/Lsurvey,pi*Nvox/Lsurvey,Nk)
test_interp_bins=np.linspace(twopi/Lsurvey,pi*Nvox/Lsurvey,2*Nk)
test_interp_bins_1=np.linspace(twopi/Lsurvey,pi*Nvox/Lsurvey,3*Nk//2)
Ptest=ktest**idx
case_names= ["unmodulated","modulated_0","modulated_1","modulated_2"]
titles=["unmodulated","modulation broad in config space","modulation medium in config space","modulation narrow in config space"]
labelsr=["","","","reconstructed"]
labelsf=["","","","fiducial"]

################################################################################################################################################################################################################
spherical_test_suite=True
if spherical_test_suite:

    # ###
    #     def __init__(self,
    #              Lxy,Lz=None,                                                            # one scaling is nonnegotiable for box->spec and spec->box calcs; the other would be useful for rectangular prism box considerations (sky plane slice is square, but LoS extent can differ)
    #              T_pristine=None,T_primary=None,P_fid=None,Nvox=None,Nvoxz=None,         # need one of either T (pristine or primary) or P to get started; I also check for any conflicts with Nvox
    #              primary_beam=None,primary_beam_args=None,primary_beam_type="Gaussian",  # primary beam considerations
    #              Nk0=10,Nk1=0,binning_mode="lin",                                        # binning considerations for power spec realizations (log mode not fully tested yet b/c not impt. for current pipeline)
    #              frac_tol=0.1,                                                           # max number of realizations
    #              k0bins_interp=None,k1bins_interp=None,                                  # bins where it would be nice to know about P_converged
    #              P_realizations=None,P_converged=None,                                   # power spectra related to averaging over those from dif box realizations
    #              verbose=False,                                                          # status updates for averaging over realizations
    #              k_fid=None,kind="cubic",avoid_extrapolation=False,                      # helper vars for converting a 1d fid power spec to a box sampling
    #              no_monopole=True,                                                       # consideration when generating boxes
    #              manual_primary_beam_modes=None,                                         # when using a discretely sampled primary beam not sampled internally using a callable, it is necessary to provide knowledge of the modes at which it was sampled
    #              ):
    # ###

    unmodulated=cosmo_stats(Lsurvey,
                            P_fid=Ptest,Nvox=Nvox,
                            Nk0=Nk,
                            k0bins_interp=test_interp_bins,k_fid=ktest)
    modulated_0=cosmo_stats(Lsurvey,
                            P_fid=Ptest,Nvox=Nvox,
                            primary_beam=Gaussian_primary,primary_beam_args=bundled0,
                            Nk0=Nk,
                            k0bins_interp=test_interp_bins,k_fid=ktest)
    modulated_1=cosmo_stats(Lsurvey,
                            P_fid=Ptest,Nvox=Nvox,
                            primary_beam=Gaussian_primary,primary_beam_args=bundled1,
                            Nk0=Nk,
                            k0bins_interp=test_interp_bins,k_fid=ktest)
    modulated_2=cosmo_stats(Lsurvey,
                            P_fid=Ptest,Nvox=Nvox,
                            primary_beam=Gaussian_primary,primary_beam_args=bundled2,
                            Nk0=Nk,
                            k0bins_interp=test_interp_bins,k_fid=ktest)
    cases=      [ unmodulated,  modulated_0,  modulated_1,  modulated_2 ]
    print("__init__ test complete")

    unmodulated.generate_box()
    modulated_0.generate_box()
    modulated_1.generate_box()
    modulated_2.generate_box()
    for i,case in enumerate(cases):
        fig,axs=plt.subplots(2,3,figsize=(8,4))
        axs[0,0].imshow(case.T_pristine[:,:,3])
        axs[0,0].set_title(case_names[i]+".T_pristine[:,:,3]")
        axs[0,1].imshow(case.T_pristine[:,37,:])
        axs[0,1].set_title(case_names[i]+".T_pristine[:,37,:]")
        axs[0,2].imshow(case.T_pristine[18,:,:])
        axs[0,2].set_title(case_names[i]+".T_pristine[18,:,:]")
        axs[1,0].imshow(case.T_primary[:,:,3])
        axs[1,0].set_title(case_names[i]+".T_primary[:,:,3]")
        axs[1,1].imshow(case.T_primary[:,37,:])
        axs[1,1].set_title(case_names[i]+".T_primary[:,37,:]")
        axs[1,2].imshow(case.T_primary[18,:,:])
        axs[1,2].set_title(case_names[i]+".T_primary[18,:,:]")
        plt.suptitle("box slices")
        plt.tight_layout()
        plt.savefig(case_names[i]+"_"+str(idx)+"_box_slices.png")
        plt.show()
    print("generate_box() test complete")

    unmodulated.generate_P()
    modulated_0.generate_P()
    modulated_1.generate_P()
    modulated_2.generate_P()
    fig,axs=plt.subplots(1,4,figsize=(15,5)) # INSPECT BINNED POWER SPEC
    for i,case in enumerate(cases):
        axs[i].plot(case.k0bins,np.array(case.P_realizations[0]).reshape(Nk,),label=labelsr[i])
        axs[i].set_title(titles[i])
        axs[i].plot(ktest,Ptest,label=labelsf[i])
        axs[i].set_xlabel("k")
        axs[i].set_ylabel("P")
    plt.legend()
    plt.suptitle("single realization reconstruction tests")
    fig.tight_layout()
    plt.savefig("test_reconstr_"+str(idx)+"_class.png")
    plt.show()
    fig,axs=plt.subplots(1,4,figsize=(15,5)) # INSPECT **UN**BINNED POWER SPEC
    for i,case in enumerate(cases):
        im=axs[i].imshow(case.unbinned_P[:,:,Nvox//2]) # is this centred the way I want it to be?
        axs[i].set_title(titles[i])
        plt.colorbar(im,ax=axs[i])
    plt.legend()
    plt.suptitle("UNBINNED - single realization reconstruction tests - slices")
    fig.tight_layout()
    plt.savefig("UNBINNED_test_reconstr_"+str(idx)+"_class_slices.png")
    plt.show()
    print("generate_P() test complete")

    unmodulated.avg_realizations()
    modulated_0.avg_realizations()
    modulated_1.avg_realizations()
    modulated_2.avg_realizations()
    fig,axs=plt.subplots(1,4,figsize=(15,5))
    for i,case in enumerate(cases):
        axs[i].plot(case.k0bins,np.array(case.P_converged),label=labelsr[i])
        axs[i].set_title(titles[i]+"\n"+str(case.realization_ceiling)+"realiz")
        axs[i].plot(ktest,Ptest,label=labelsf[i])
        axs[i].set_xlabel("k")
        axs[i].set_ylabel("P")
    plt.legend()
    plt.suptitle("converged reconstruction tests")
    fig.tight_layout()
    plt.savefig("avg_realizations_"+str(idx)+"_class.png")
    plt.show()
    print("avg_realizations() test complete")

    unmodulated.interpolate_P()
    modulated_0.interpolate_P()
    modulated_1.interpolate_P()
    modulated_2.interpolate_P()
    print("unmodulated.realization_ceiling,modulated_0.realization_ceiling,modulated_1.realization_ceiling,modulated_2.num_z_evaled=",unmodulated.realization_ceiling,modulated_0.realization_ceiling,modulated_1.realization_ceiling,modulated_2.realization_ceiling)
    labelsi=["","","","interpolated"]
    fig,axs=plt.subplots(1,4,figsize=(15,5))
    for i,case in enumerate(cases):
        axs[i].scatter(case.k0bins,       np.array(case.P_converged),label=labelsr[i])
        axs[i].scatter(case.k0bins_interp,np.array(case.P_interp),   label=labelsi[i])
        axs[i].set_title(titles[i])
        axs[i].plot(ktest,Ptest,label=labelsf[i])
        axs[i].set_xlabel("k")
        axs[i].set_ylabel("P")
    plt.legend()
    plt.suptitle("interpolated converged reconstruction tests")
    fig.tight_layout()
    plt.savefig("interp_"+str(idx)+"_class.png")
    plt.show()
    print("interpolate_P() test complete")

################################################################################################################################################################################################################
cylindrical_test_suite=True
if cylindrical_test_suite:
    unmodulated=cosmo_stats(Lsurvey,
                            P_fid=Ptest,Nvox=Nvox,
                            Nk0=Nk,Nk1=Nk1,
                            k0bins_interp=test_interp_bins,
                            k1bins_interp=test_interp_bins_1,k_fid=ktest)
    modulated_0=cosmo_stats(Lsurvey,
                            P_fid=Ptest,Nvox=Nvox,
                            primary_beam=Gaussian_primary,primary_beam_args=bundled0,
                            Nk0=Nk,Nk1=Nk1,
                            k0bins_interp=test_interp_bins,
                            k1bins_interp=test_interp_bins_1,k_fid=ktest)
    modulated_1=cosmo_stats(Lsurvey,
                            P_fid=Ptest,Nvox=Nvox,
                            primary_beam=Gaussian_primary,primary_beam_args=bundled1,
                            Nk0=Nk,Nk1=Nk1,
                            k0bins_interp=test_interp_bins,
                            k1bins_interp=test_interp_bins_1,k_fid=ktest)
    modulated_2=cosmo_stats(Lsurvey,
                            P_fid=Ptest,Nvox=Nvox,
                            primary_beam=Gaussian_primary,primary_beam_args=bundled2,
                            Nk0=Nk,Nk1=Nk1,
                            k0bins_interp=test_interp_bins,
                            k1bins_interp=test_interp_bins_1,k_fid=ktest)
    
    cases=      [ unmodulated,  modulated_0,  modulated_1,  modulated_2 ]
    print("__init__ test complete")

    unmodulated.generate_box()
    modulated_0.generate_box()
    modulated_1.generate_box()
    modulated_2.generate_box()
    for i,case in enumerate(cases):
        fig,axs=plt.subplots(2,3,figsize=(8,4))
        axs[0,0].imshow(case.T_pristine[:,:,3])
        axs[0,0].set_title(case_names[i]+".T_pristine[:,:,3]")
        axs[0,1].imshow(case.T_pristine[:,37,:])
        axs[0,1].set_title(case_names[i]+".T_pristine[:,37,:]")
        axs[0,2].imshow(case.T_pristine[18,:,:])
        axs[0,2].set_title(case_names[i]+".T_pristine[18,:,:]")
        axs[1,0].imshow(case.T_primary[:,:,3])
        axs[1,0].set_title(case_names[i]+".T_primary[:,:,3]")
        axs[1,1].imshow(case.T_primary[:,37,:])
        axs[1,1].set_title(case_names[i]+".T_primary[:,37,:]")
        axs[1,2].imshow(case.T_primary[18,:,:])
        axs[1,2].set_title(case_names[i]+".T_primary[18,:,:]")
        plt.suptitle("box slices")
        plt.tight_layout()
        plt.savefig("CYL_BRANCH_"+case_names[i]+"_"+str(idx)+"_box_slices.png")
        plt.show()
    print("generate_box() test complete") # should be just as trivial as it was last time... the only difference from the sph case is that the class instances were initialized differenltly

    unmodulated.generate_P()
    modulated_0.generate_P()
    modulated_1.generate_P()
    modulated_2.generate_P()
    fig,axs=plt.subplots(1,4,figsize=(15,5))
    for i,case in enumerate(cases):
        im=axs[i].imshow(np.array(case.P_realizations[0]).reshape(Nk,Nk1))
        plt.colorbar(im,ax=axs[i])
        axs[i].set_title(titles[i])
        axs[i].set_ylabel("k$_{||}$ index")
        axs[i].set_xlabel("k$_\perp$ index")
    plt.legend()
    plt.suptitle("converged reconstruction tests")
    fig.tight_layout()
    plt.savefig("CYL_BRANCH_avg_realizations_"+str(idx)+"_class.png")
    plt.show()
    print("generate_P() test complete")

    unmodulated.avg_realizations()
    modulated_0.avg_realizations()
    modulated_1.avg_realizations()
    modulated_2.avg_realizations()
    fig,axs=plt.subplots(1,4,figsize=(15,5))
    for i,case in enumerate(cases):
        im=axs[i].imshow(case.P_converged)
        plt.colorbar(im,ax=axs[i])
        axs[i].set_title(titles[i]+"\n"+str(case.realization_ceiling)+"realiz")
        axs[i].set_ylabel("k$_{||}$ index")
        axs[i].set_xlabel("k$_\perp$ index")
    plt.legend()
    plt.suptitle("converged reconstruction tests")
    fig.tight_layout()
    plt.savefig("CYL_BRANCH_avg_realizations_"+str(idx)+"_class.png")
    plt.show()
    print("avg_realizations() test complete")

    unmodulated.interpolate_P()
    modulated_0.interpolate_P()
    modulated_1.interpolate_P()
    modulated_2.interpolate_P()
    fig,axs=plt.subplots(1,4,figsize=(15,5))
    for i,case in enumerate(cases):
        im=axs[i].imshow(case.P_interp)
        plt.colorbar(im,ax=axs[i])
        axs[i].set_title(titles[i])
        axs[i].set_ylabel("k$_{||}$ index")
        axs[i].set_xlabel("k$_\perp$ index")
    plt.legend()
    plt.suptitle("interpolated converged reconstruction tests")
    fig.tight_layout()
    plt.savefig("CYL_BRANCH_interp_"+str(idx)+"_class.png")
    plt.show()
    print("interpolate_P() test complete")