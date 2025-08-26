import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interpn

def stack_to_box(light_cone,
                 lc_xy_bins,lc_z_bins,
                 interp_method="cubic",avoid_extrap=False,fill_value=None):
    """
    light_cone    :: (Nx_lc,Ny_lc,Nz_lc) of floats :: lc from stacking dirty images         :: arbitrary (do some better documentation of the relative scaling thing)
    Lxy_lc_low_z  :: float                         :: sky plane lc extent of lowest-z slice :: Mpc
    Lz_lc         :: float                         :: LoS lc extent                         :: Mpc
    lc_xy_bins    :: (Nxy,Nz) of floats            :: 1d arr of transv. bin fl. per slice   :: Mpc
    lc_z_bins     :: (Nz,) of floats               :: 1d arr of comoving dists. per slice   :: Mpc
    interp_method :: str                           :: interpolation method for interpn      :: ---
    avoid_extrap  :: bool                          :: y/n avoid extrapolation in interpn    :: ---
    fill_value    :: NoneType/float                :: what to do about extrapolated pts     :: ---/Mpc
    """
    Nz_lc=light_cone.shape[-1]
    Lxy_lc_low_z=lc_xy_bins[-1,0]-lc_xy_bins[0,0]
    Lz_lc=lc_z_bins[-1]-lc_z_bins[0]
    Delta=Lz_lc/Nz_lc
    Nxy=int(Lxy_lc_low_z/Delta)
    xy_ref=lc_xy_bins[0,0]
    box_xy=xy_ref+np.linspace(0,Lxy_lc_low_z,Nxy) # this one is dif, but box_z is the same as lc_z
    box_xx,box_yy=np.meshgrid(box_xy,box_xy,indexing="ij")

    box=np.zeros((Nxy,Nxy,Nz_lc))
    for i in range(Nz_lc):
        lc_xy_bins_slice=lc_xy_bins[:,i]
        box[:,:,i]=interpn((lc_xy_bins_slice,lc_xy_bins_slice),light_cone[:,:,i],(box_xx,box_yy),method=interp_method,bounds_error=avoid_extrap,fill_value=fill_value)
    
    return box,box_xy

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