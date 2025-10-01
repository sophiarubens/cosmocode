import numpy as np
from matplotlib import pyplot as plt
from CHORD_vis import *

############################################################################################################################################################################################################################
"""
SHORTHAND:
fidu = neither antenna nor primary beam perturbations (fiducial array)
antp = antenna perturbations, but no primary beam perturbations
prbp = no antenna perturbations, but yes primary beam perturbations
both = both antenna and primary beam perturbations
"""

N_ant_to_pert=100
N_pbs_to_pert=100
N_pert_types=4
fidu=CHORD_image(                                   N_pert_types=0)
antp=CHORD_image(num_ant_pos_to_pert=N_ant_to_pert, N_pert_types=0)
prbp=CHORD_image(num_pbws_to_pert=N_pbs_to_pert,    N_pert_types=N_pert_types)
both=CHORD_image(num_ant_pos_to_pert=N_ant_to_pert,
                 num_pbws_to_pert=N_pbs_to_pert,    N_pert_types=N_pert_types)
N_hr_angles=fidu.num_timesteps
N_obs_hrs=fidu.num_hrs
ant_pert=fidu.ant_pos_pert_sigma
prb_pert=fidu.pbw_pert_sigma

print("N_pert_types=",N_pert_types)
Npix=4096
fidu.calc_dirty_image(Npix=Npix)
antp.calc_dirty_image(Npix=Npix)
prbp.calc_dirty_image(Npix=Npix)
both.calc_dirty_image(Npix=Npix)

colours_b=plt.cm.Blues( np.linspace(1,0.2,N_hr_angles))
test_freq=885
lambda_obs=c/(test_freq*1e6)
z_obs=nu_HI_z0/test_freq-1.

cases=[fidu,antp,prbp,both]

# plot
dotsize=1
fig,axs=plt.subplots(4,7,figsize=(31,15))
for i in range(4):
    axs[i,0].set_xlabel("E (m)")
    axs[i,0].set_ylabel("N (m)")
    axs[i,1].set_xlabel("u (m)")
    axs[i,1].set_ylabel("v (m)")

    for j in range(2,4):
        axs[i,j].set_xlabel("u ($\lambda$)")
        axs[i,j].set_ylabel("v ($\lambda$)")

    for j in range(4,7):
        axs[i,j].set_xlabel("$θ_x$ (rad)")
        axs[i,j].set_ylabel("$θ_y$ (rad)")
axs[0,0].set_title("oversimplified array layout\n (no receiver hut holes,\n eyeballed array rotation and elevation,\n colour ~ relative U-coord)\nFIDUCIAL ARRAY")
axs[0,1].set_title("instantaneous uv-coverage/\ndirty beam")
axs[0,2].set_title(str(round(N_obs_hrs,3))+"-hr rotation-synthesized uv-coverage\nsampled every "+str(round(3600/(N_hr_angles/N_obs_hrs)))+" s (colour ~ baseline)")
axs[0,3].set_title("binned rot-synth and\n primary-beamed uv")
axs[0,4].set_title("dirty image\n(IFT(gridded uv) \n"+str(1024)+" bins/axis)")
axs[1,0].set_title("PERTURBED ANTENNA POSITIONS\nperturbation magnitude="+str(ant_pert*1e3)+"mm")
axs[1,5].set_title("ratio: \nfiducial/perturbed")
axs[1,6].set_title("residual: \nfiducial-perturbed")
axs[2,0].set_title("PERTURBED PRIMARY BEAM WIDTHS\nfractional perturbation magnitude="+str(prb_pert))
axs[3,0].set_title("PERTURBED ANTENNA POSITIONS\n AND PRIMARY BEAM WIDTHS\ncombined effects of the above cases")

off=0.8 # multiplicative offset so the rms isn't squished against the edge of each annotated subplot
for k,case in enumerate(cases):
    axs[k,0].scatter(case.antennas_xyz[:,0],case.antennas_xyz[:,1],s=dotsize,c=case.antennas_xyz[:,2],cmap=trunc_Blues)
    axs[k,1].scatter(case.uvw_inst[:,0],case.uvw_inst[:,1],s=dotsize)

    for i in range(N_hr_angles):
        axs[k,2].scatter(case.uv_synth[:,0,i],case.uv_synth[:,1,i],color=colours_b[i],s=dotsize) # all baselines, x/y coord, ith time step //one colour = one instance of instantaneous uv-coverage
    uvplane=case.uvplane
    im=axs[k,3].imshow(uvplane,cmap="Blues",vmin=np.percentile(uvplane,1),vmax=np.percentile(uvplane,99),origin="lower",
                       extent=[case.uvmin,case.uvmax,case.uvmin,case.uvmax])
    clb=plt.colorbar(im,ax=axs[k,3])
    clb.ax.set_title("#bl")

    dirty_image=case.dirty_image
    if (k==0):
        dirty_image_fidu=dirty_image
    thetalim=case.thetamax
    theta_extent=[-thetalim,thetalim,-thetalim,thetalim]
    im=axs[k,4].imshow(dirty_image,cmap="Blues",vmin=np.percentile(dirty_image,2),vmax=np.percentile(dirty_image,98),origin="lower",
                       extent=theta_extent)
    plt.colorbar(im,ax=axs[k,4])

    if (k>0):
        ratio=dirty_image_fidu/dirty_image
        im=axs[k,5].imshow(ratio,cmap="Blues",origin="lower",vmin=np.nanpercentile(ratio,2),vmax=np.nanpercentile(ratio,98),extent=theta_extent)
        plt.colorbar(im,ax=axs[k,5])
        axs[k,5].text(-off*thetalim,-off*thetalim,"rms={:8.4}".format(np.sqrt(np.nanmean(ratio**2))),c="r")

        residual=dirty_image_fidu-dirty_image
        im=axs[k,6].imshow(residual,cmap="Blues",origin="lower",vmin=np.percentile(residual,2),vmax=np.percentile(residual,98),extent=theta_extent)
        plt.colorbar(im,ax=axs[k,6])
        axs[k,6].text(-off*thetalim,-off*thetalim,"rms={:8.4}".format(np.sqrt(np.nanmean(residual**2))),c="r")

        if (k==2): # save the ratios and residuals from the "primary beam width perturbations only" case to use later
            np.save("ratio_"+str(test_freq)+"_MHz_"+str(N_pert_types)+"_pbw_pert_types.npy",ratio) # not featured in file name: 100 perturbed antennas, 1e-2 perturbation magnitude, etc.
            np.save("resid_"+str(test_freq)+"_MHz_"+str(N_pert_types)+"_pbw_pert_types.npy",residual)

plt.suptitle("simulated CHORD-512 observing at "+str(int(test_freq))+" MHz (z="+str(round(z_obs,3))+") with "+str(N_pert_types)+" kinds of primary beam perturbations")
plt.tight_layout()
plt.savefig("simulated_CHORD_512_"+str(int(test_freq))+"_MHz_"+str(int(ant_pert*1e3))+"_mm_"+str(int(N_ant_to_pert))+"_ant.png",dpi=200)
plt.show()