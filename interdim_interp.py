import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d,interpn
from power_class import *

pi=np.pi
twopi=2.*pi

L=126
N=53
Nk_fid=13
# idx=-0.1
idx=2.3
k_fid=np.linspace(twopi/L,pi*N/L,Nk_fid)           # know about the power spec at these modes
P_fid=k_fid**idx

# # testing approach before adding to class
# cstest=cosmo_stats(L,Nvox=N,P_fid=P_fid)
# kbox_centre=cstest.k_grid_centre # for generate_P
# kbox_corner=cstest.k_grid_corner # for generate_box # want to know about the power spec at these modes

# kbox_corner_flat=np.reshape(kbox_corner,(N**3,))

# P_fid_interpolator=interp1d(k_fid,P_fid,kind="cubic",bounds_error=False,fill_value="extrapolate")
# P_interp_flat=P_fid_interpolator(kbox_corner_flat)
# P_interp=np.reshape(P_interp_flat,(N,N,N))

# testing the version that made it into the class
cstest=cosmo_stats(L,Nvox=N,P_fid=P_fid,k_fid=k_fid)
P_interp=cstest.P_fid_box

plt.figure()
plt.plot(k_fid,P_fid)
plt.xlabel("k")
plt.ylabel("P")
plt.title("fiducial 1d power")
plt.savefig("P_fid.png")
plt.show()

fig,axs=plt.subplots(3,4)
vn=np.min(P_interp)
vx=np.max(P_interp)
for j in range(4):
    frac=j*N//3
    if frac==N:
        frac-=1
    im=axs[0,j].imshow(P_interp[frac,:,:],vmin=vn,vmax=vx)
    axs[0,j].set_xlabel("k_y")
    axs[0,j].set_ylabel("k_z")
    axs[0,j].set_title("["+str(frac)+",:,:]")
    im=axs[1,j].imshow(P_interp[:,frac,:],vmin=vn,vmax=vx)
    axs[1,j].set_xlabel("k_x")
    axs[1,j].set_ylabel("k_z")
    axs[1,j].set_title("[:,"+str(frac)+",:]")
    im=axs[2,j].imshow(P_interp[:,:,frac],vmin=vn,vmax=vx)
    axs[2,j].set_xlabel("k_x")
    axs[2,j].set_ylabel("k_y")
    axs[2,j].set_title("[:,:,"+str(frac)+"]")
fig.colorbar(im)
plt.suptitle("slices of grid-interpolated P_fid")
plt.tight_layout()
plt.savefig("slices_of_grid_interpolated_P_fid.png")
plt.show()
