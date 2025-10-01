import numpy as np
from matplotlib import pyplot as plt
from CHORD_vis import *

############################################################################################################################################################################################################################
"""
SHORTHAND:
fidu = neither antenna nor primary beam perturbations (fiducial array)
prbp = no antenna position perturbations, but yes primary beam perturbations
"""

N_ant_to_pert=100
N_pbs_to_pert=100
N_pert_types=4
fidu=CHORD_image(                                   N_pert_types=0)
prbp=CHORD_image(num_pbws_to_pert=N_pbs_to_pert,    N_pert_types=N_pert_types)
N_hr_angles=fidu.num_timesteps
N_obs_hrs=fidu.num_hrs
ant_pert=fidu.ant_pos_pert_sigma
prb_pert=fidu.pbw_pert_sigma

print("N_pert_types=",N_pert_types)
Npix=2048
fidu.calc_dirty_image(Npix=Npix)
prbp.calc_dirty_image(Npix=Npix)

colours_b=plt.cm.Blues( np.linspace(1,0.2,N_hr_angles))
test_freq=885
lambda_obs=c/(test_freq*1e6)
z_obs=nu_HI_z0/test_freq-1.

cases=[fidu,prbp]

# plot
dotsize=1
fig,axs=plt.subplots(2,4,figsize=(18,8))
for i in range(2):
    axs[i,0].set_xlabel("u ($\lambda$)")
    axs[i,0].set_ylabel("v ($\lambda$)")

    for j in range(1,4):
        axs[i,j].set_xlabel("$θ_x$ (rad)")
        axs[i,j].set_ylabel("$θ_y$ (rad)")
axs[0,0].set_title("binned rot-synth and\n primary-beamed uv")
axs[0,1].set_title("dirty image\n(IFT(gridded uv) \n"+str(Npix)+" bins/axis)")
axs[1,2].set_title("ratio: \nfiducial/perturbed")
axs[1,3].set_title("residual: \nfiducial-perturbed")
axs[1,0].set_title("PERTURBED PBWs\nfractional magnitude="+str(prb_pert))

off=0.8 # multiplicative offset so the rms isn't squished against the edge of each annotated subplot
for k,case in enumerate(cases):
    uvplane=case.uvplane
    im=axs[k,0].imshow(uvplane,cmap="Blues",vmin=np.percentile(uvplane,1),vmax=np.percentile(uvplane,99),origin="lower",
                       extent=[case.uvmin,case.uvmax,case.uvmin,case.uvmax])
    clb=plt.colorbar(im,ax=axs[k,0])
    clb.ax.set_title("#bl")

    dirty_image=case.dirty_image
    if (k==0):
        dirty_image_fidu=dirty_image
    thetalim=case.thetamax
    theta_extent=[-thetalim,thetalim,-thetalim,thetalim]
    im=axs[k,1].imshow(dirty_image,cmap="Blues",vmin=np.percentile(dirty_image,2),vmax=np.percentile(dirty_image,98),origin="lower",
                       extent=theta_extent)
    plt.colorbar(im,ax=axs[k,1])

    if (k>0):
        ratio=dirty_image_fidu/dirty_image
        im=axs[k,2].imshow(ratio,cmap="Blues",origin="lower",vmin=np.nanpercentile(ratio,2),vmax=np.nanpercentile(ratio,98),extent=theta_extent)
        plt.colorbar(im,ax=axs[k,2])
        axs[k,2].text(-off*thetalim,-off*thetalim,"rms={:8.4}".format(np.sqrt(np.nanmean(ratio**2))),c="r")

        residual=dirty_image_fidu-dirty_image
        im=axs[k,3].imshow(residual,cmap="Blues",origin="lower",vmin=np.percentile(residual,2),vmax=np.percentile(residual,98),extent=theta_extent)
        plt.colorbar(im,ax=axs[k,3])
        axs[k,3].text(-off*thetalim,-off*thetalim,"rms={:8.4}".format(np.sqrt(np.nanmean(residual**2))),c="r")

plt.suptitle("simulated CHORD-512 observing at "+str(int(test_freq))+" MHz (z="+str(round(z_obs,3))+") with "+str(N_pert_types)+" kinds of primary beam perturbations")
plt.tight_layout()
plt.savefig("simulated_CHORD_512_"+str(int(test_freq))+"_MHz_"+str(int(ant_pert*1e3))+"_mm_"+str(int(N_ant_to_pert))+"_ant_"+str(Npix)+".png",dpi=200)
plt.show()