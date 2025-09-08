import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interpn
from CHORD_vis import *
from cosmo_distances import *

def stack_to_box(N_NS,N_EW,offset_deg,num_antpos_to_perturb,antpos_perturbation_sigma,num_pbs_to_perturb,pb_perturbation_sigma,observatory_latitude,
                 survey_freqs,fiducial_beam_args,
                 num_hrs=1./2.,num_timesteps=15,dec=30.,
                 nbins_coarse=32,nbins=1024):
    
    perp_fwhm,par_fwhm=fiducial_beam_args # how to unpack for now, given that I'm not back to elliptical beams over here
    antennas_xyz,ant_pb_frac_widths,_=CHORD_antenna_positions(N_NS=N_NS,N_EW=N_EW,offset_deg=offset_deg,                                                        # basics required to specify a CHORD-like array
                                                               num_antpos_to_perturb=num_antpos_to_perturb,antpos_perturbation_sigma=antpos_perturbation_sigma, # controls for examining the effect of misplaced antennas
                                                               num_pbs_to_perturb=num_pbs_to_perturb,pb_perturbation_sigma=pb_perturbation_sigma,               # controls for examining the effect of primary beam width mischaracterizations on a per-antenna basis
                                                               observatory_latitude=observatory_latitude)                                                       # final non-stored arg is [indices_ants,indices_pbs]
    uvw=calc_inst_uvw(antennas_xyz,ant_pb_frac_widths,N_NS=N_NS,N_EW=N_EW)
    uv_synth=calc_rot_synth_uv(uvw,lambda_obs=nu_HI_z0,num_hrs=num_hrs,num_timesteps=num_timesteps,dec=dec)

    dirty_image_0,_,uv_bin_edges,_=calc_dirty_image(uv_synth,ant_pb_frac_widths,perp_fwhm,nbins_coarse=nbins_coarse,nbins=nbins) # skipped args are, respectively, uvplane and theta_lims
    uv_bin_edges_0=np.copy(uv_bin_edges) # modes to interpolate to= modes from the base of the light cone
    uu_bin_edges_0,vv_bin_edges_0=np.meshgrid(uv_bin_edges_0,uv_bin_edges_0,indexing="ij")
    N_chans=len(survey_freqs)

    # oversimplified case where I ignore beam chromaticity over the survey bandwidth
    light_cone=np.zeros((nbins,nbins,N_chans))
    lambda_0=c/(survey_freqs[0]*1e6)
    survey_Dc=0.*survey_freqs
    for i,ctr_freq in enumerate(survey_freqs):
        survey_Dc[i]=comoving_distance(freq2z(ctr_freq))
        lambda_curr=c/(survey_freqs[i]*1e6)
        uv_bin_edges=uv_bin_edges_0*lambda_0/lambda_curr # rescale the uv-plane as appropriate
        LoS_modulation=np.exp(-survey_Dc[i]**2/(2*sigma_LoS**2))
        dirty_image=dirty_image_0*LoS_modulation
        interpolated_slice=interpn((uv_bin_edges,uv_bin_edges),dirty_image,(uu_bin_edges_0,vv_bin_edges_0)) # this takes care of the chunk excision and interpolation in one step... but only for the current slice
        light_cone[:,:,i]=interpolated_slice

    return light_cone,survey_Dc

# test: create something (Nxy,Nxy,Nz) but that, if you were to plot it on a uniform grid, would be a truncated pyramid/ have light cone shape
Nxy=7
Nz=8
xy_half=3.52
z_range=9.12
z_lo=1.6
lc_z_bins=z_lo+np.linspace(0,z_range,Nz)
lc_multiplier=1.1
lc_xy_bins=np.zeros((Nxy,Nz))
lc_vals=np.zeros((Nxy,Nxy,Nz))
for i in range(Nz):
    lc_xy_bins_i=lc_multiplier**i*np.linspace(-xy_half,xy_half,Nxy)
    lc_xy_bins[:,i]=lc_xy_bins_i
    lc_xy_grid_x_i,lc_xy_grid_y_i=np.meshgrid(lc_xy_bins_i,lc_xy_bins_i,indexing="ij")
    lc_vals[:,:,i]=np.exp(-(lc_xy_grid_x_i**2+lc_xy_grid_y_i**2)/2)

box,box_xy=stack_to_box(lc_vals,lc_xy_bins,lc_z_bins)

# fig,axs=plt.subplots(1,3)
# im=axs[0].imshow(box[0,:,:],origin="lower")
# plt.colorbar(im,ax=axs[0])
# im=axs[1].imshow(box[:,0,:],origin="lower")
# plt.colorbar(im,ax=axs[1])
# im=axs[2].imshow(box[:,:,0],origin="lower")
# plt.colorbar(im,ax=axs[2])
# plt.show()

fig,axs=plt.subplots(2,Nz,figsize=(25,7))
title_prefix0=["light cone","","","","","","",""]
title_prefix1=["interpolated box","","","","","","",""]
for j in range(Nz):
    im=axs[0,j].imshow(lc_vals[:,:,j],origin="lower",extent=[lc_xy_bins[0,j],lc_xy_bins[-1,j],lc_xy_bins[0,j],lc_xy_bins[-1,j]])
    plt.colorbar(im,ax=axs[0,j])
    im=axs[1,j].imshow(box[:,:,j],    origin="lower",extent=[box_xy[0],box_xy[-1],box_xy[0],box_xy[-1]])
    plt.colorbar(im,ax=axs[1,j])
    for i in range(2):
        axs[i,j].set_xlabel("x (Mpc)")
        axs[i,j].set_ylabel("y (Mpc)")
    axs[0,j].set_title(title_prefix0[j]+" "+str(round(lc_z_bins[j],2)))
    axs[1,j].set_title(title_prefix1[j]+" "+str(round(lc_z_bins[j],2)))
plt.suptitle("slices before/after LC->box interpolation")
plt.savefig("inspect_box_conversion.png")
plt.tight_layout()
plt.show()