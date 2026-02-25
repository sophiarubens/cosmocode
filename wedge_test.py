import numpy as np
from matplotlib import pyplot as plt
from forecasting_pipeline import *

"""
cosmo_stats(object):
    def __init__(self,
                 Lxy,Lz=None,                                                                       # one scaling is nonnegotiable for box->spec and spec->box calcs; the other would be useful for rectangular prism box considerations (sky plane slice is square, but LoS extent can differ)
                 T_pristine=None,T_primary=None,P_fid=None,Nvox=None,Nvoxz=None,                    # need one of either T (pristine or primary) or P to get started; I also check for any conflicts with Nvox
                 primary_beam_num=None,primary_beam_aux_num=None, primary_beam_type_num="Gaussian", # primary beam considerations
                 primary_beam_den=None,primary_beam_aux_den=None, primary_beam_type_den="Gaussian", # systematic-y beam (optional)
                 Nk0=10,Nk1=0,binning_mode="lin",bin_each_realization=False,                        # binning considerations for power spec realizations (log mode not fully tested yet b/c not impt. for current pipeline)
                 frac_tol=0.1,                                                                      # max number of realizations
                 k0bins_interp=None,k1bins_interp=None,                                             # bins where it would be nice to know about P_converged
                 P_converged=None,verbose=False,                                                    # status updates for averaging over realizations
                 k_fid=None,kind="cubic",avoid_extrapolation=False,                                 # helper vars for converting a 1d fid power spec to a box sampling
                 no_monopole=True,                                                                  # consideration when generating boxes
                 manual_primary_beam_modes=None,                                                    # when using a discretely sampled primary beam not sampled internally using a callable, it is necessary to provide knowledge of the modes at which it was sampled
                 radial_taper=None,image_taper=None,                                                # implement soon: quick way to use an Airy beam in per-antenna mode
                 wedge_cut=False,nu_ctr_for_wedge=None):
    """

Lxy=123
Lz=88
Nvox=80

# wedge_cut=False
# wedge_cut=True,nu_ctr_for_wedge=800

kmax=twopi/(Lxy/Nvox)
N_fid_k=546
k_fid=np.linspace(0,1.5*kmax,N_fid_k)
k_idx=np.arange(0,N_fid_k,1)
P_fid=np.exp(-(k_idx-45)**2/50)
plt.figure()
plt.plot(k_fid,P_fid)
plt.xlabel("k (1/Mpc)")
plt.ylabel("power (K^2 Mpc^3)")
plt.title("Toy power spectrum to generate realizations from")
plt.savefig("toy_power_spec.png")

wedge_agnostic=cosmo_stats(Lxy,P_fid=P_fid,k_fid=k_fid,
                           Nvox=Nvox)
wedge_agnostic.generate_box()
wedge_agnostic_box=wedge_agnostic.T_pristine
xy_vec=wedge_agnostic.xy_vec_for_box
z_vec=wedge_agnostic.z_vec_for_box
wedge_aware=   cosmo_stats(Lxy,P_fid=P_fid,k_fid=k_fid,
                           Nvox=Nvox,wedge_cut=True,nu_ctr_for_wedge=800.)
wedge_aware.generate_box()
wedge_aware_box=wedge_aware.T_pristine

boxes=[wedge_agnostic_box,wedge_aware_box]
box_names=["wedge-agnostic box","wedge-aware box"]

N_slices=4
for i,box in enumerate(boxes):
    N0,N1,N2=box.shape
    fig,axs=plt.subplots(N_slices,3,figsize=(12,12),layout="constrained")
    for j in range(N_slices):
        cut0=int(N0*j/4)
        yz_slice=box[cut0,:,:]
        im=axs[j,0].imshow(yz_slice.T,origin="lower", #vmax=1,
                        extent=[xy_vec[0],xy_vec[-1],z_vec[0],z_vec[-1]])
        plt.colorbar(im,ax=axs[j,0])
        axs[j,0].set_xlabel("y (Mpc)")
        axs[j,0].set_ylabel("z (Mpc)")
        axs[j,0].set_title(str(cut0)+"/"+str(N0)+" yz")

        cut1=int(N1*j/4)
        xz_slice=box[:,cut1,:]
        im=axs[j,1].imshow(xz_slice.T,origin="lower", #vmax=1,
                        extent=[xy_vec[0],xy_vec[-1],z_vec[0],z_vec[-1]])
        plt.colorbar(im,ax=axs[j,1])
        axs[j,1].set_xlabel("x (Mpc)")
        axs[j,1].set_ylabel("z (Mpc)")
        axs[j,1].set_title(str(cut1)+"/"+str(N1)+" xz")

        cut2=int(N2*j/4)
        xy_slice=box[:,:,cut2]
        im=axs[j,2].imshow(xy_slice.T,origin="lower", #vmax=1,
                        extent=[xy_vec[0],xy_vec[-1],xy_vec[0],xy_vec[-1]])
        plt.colorbar(im,ax=axs[j,2])
        axs[j,2].set_xlabel("x (Mpc)")
        axs[j,2].set_ylabel("y (Mpc)")
        axs[j,2].set_title(str(cut2)+"/"+str(N2)+" xy")
    plt.savefig(box_names[i]+".png",dpi=800)