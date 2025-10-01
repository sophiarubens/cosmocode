import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import Blues
from CHORD_vis import *
from cosmo_distances import *
import time

test_freq=400
lambda_obs=c/(test_freq*1e6)
nu_HI_z0=1420.405751768 # MHz
z_obs=nu_HI_z0/test_freq-1.
deltanu=0.183

fidu_400=CHORD_image(nu_ctr=400., N_pert_types=0)
fidu_900=CHORD_image(nu_ctr=900., N_pert_types=0)
dimg400,uv_bin_edges400,thetamax400=fidu_400.calc_dirty_image()
dimg900,uv_bin_edges900,thetamax900=fidu_900.calc_dirty_image()
print("400 MHz dirty image peak is",np.max(dimg400))
print("900 MHz dirty image peak is",np.max(dimg900))

print("thetamax400,thetamax900=",thetamax400,thetamax900)

residual=dimg400-dimg900
ratio=dimg400/dimg900

# not using extent keyword to set physical theta lims b/c I'm hackily acting like they're on the same scale
fig,axs=plt.subplots(2,2,figsize=(10,8))
im=axs[0,0].imshow(dimg400,origin="lower",cmap=Blues, vmin=np.percentile(dimg400,2),vmax=np.percentile(dimg400,98))
plt.colorbar(im,ax=axs[0,0])
axs[0,0].set_title("400 MHz beam width")
im=axs[0,1].imshow(dimg900,origin="lower",cmap=Blues, vmin=np.percentile(dimg900,2),vmax=np.percentile(dimg900,98))
plt.colorbar(im,ax=axs[0,1])
axs[0,1].set_title("900 MHz beam width")
im=axs[1,0].imshow(residual,origin="lower",cmap=Blues, vmin=np.percentile(residual,2),vmax=np.percentile(residual,98))
plt.colorbar(im,ax=axs[1,0])
axs[1,0].set_title("residual 400-900")
im=axs[1,1].imshow(ratio,origin="lower",cmap=Blues, vmin=np.percentile(ratio,2),vmax=np.percentile(ratio,98))
plt.colorbar(im,ax=axs[1,1])
axs[1,1].set_title("ratio 400/900")

for i in range(2):
    for j in range(2):
        axs[i,j].set_xlabel("θx index")
        axs[i,j].set_ylabel("θy index")
plt.suptitle("0 perturbed antenna dirty image checks")
plt.tight_layout()
plt.savefig("zero_pert_checks.png")
plt.show()

# pert=CHORD_image(nu_ctr=test_freq, N_pert_types=2,  num_pbws_to_pert=100)