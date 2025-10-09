import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import Blues
from cosmo_distances import *

fidu_images=np.load("fidu_box_363_256.npy")
pert_images=np.load("pert_box_363_256.npy")
statuses=[fidu_images,pert_images]
status_names=["fiducial","perturbed"]

nu_ctr=363.
nu_HI_z0=1420.405751768 # MHz
bw=nu_ctr/15. # reverse-engineering the class calculation for the call used when creating that box
N_chan=int(bw/0.183)
nu_lo=nu_ctr-bw/2.
nu_hi=nu_ctr+bw/2.
surv_channels=np.linspace(nu_lo,nu_hi,N_chan)
surv_z=(nu_HI_z0-surv_channels)/nu_HI_z0
Dc=np.array([comoving_distance(z) for z in surv_z])

fpct2=np.percentile(fidu_images,2)
fpct98=np.percentile(fidu_images,98)
ppct2=np.percentile(pert_images,2)
ppct98=np.percentile(pert_images,98)
pct2s=[fpct2,ppct2]
pct98s=[fpct98,ppct98]
cases=[0,-1]
thetamax=2.3466116336602156 # carried over literally from what I calculated in the other script... 06 Oct. ~17:30 run
ex=[-thetamax,thetamax,-thetamax,thetamax]
line=1

fig,axs=plt.subplots(2,3,figsize=(16,12))
for i,stat in enumerate(statuses):
    for j,case in enumerate(cases):
        axs[i,j].imshow(stat[case], origin="lower", vmin=pct2s[i],vmax=pct98s[i], cmap=Blues, extent=ex)
        if (j==0):
            prefix=status_names[i]+" beams\n"
        else:
            prefix=""
        axs[i,j].set_title(prefix+"slice {:2}: freq={:8.5} MHz; Dc={:8.5} Mpc".format(case,surv_channels[case],Dc[case]))
    ratio_edges=stat[0]/stat[-1]
    axs[i,2].imshow(ratio_edges,origin="lower", vmin=np.percentile(ratio_edges,2),vmax=np.percentile(ratio_edges,98), cmap=Blues, extent=ex)
    axs[i,2].set_title("slice ratio closest/farthest")
    for k in range(3):
        axs[i,k].set_xlabel("θx")
        axs[i,k].set_ylabel("θy")
        axs[i,k].axvline(line,c="C2")
        axs[i,k].axvline(-line,c="C2")
        axs[i,k].axhline(line,c="C2")
        axs[i,k].axhline(-line,c="C2")
plt.suptitle("verifying image stacking scaling intuition—"+str(int(nu_ctr))+" MHz survey")
plt.savefig("verifying_stacking_scaling"+str(int(nu_ctr))+".png")
plt.show()

# Nrow=12
# Ncol=13
# fig,axs = plt.subplots(Nrow,Ncol,figsize=(25,25))
# for i in range(Nrow):
#     for j in range(Ncol):
#         k=i*Nrow+j
#         axs[i,j].imshow(images[k],origin="lower",cmap=Blues,vmin=pct2,vmax=pct98)
#         axs[i,j].set_xlabel("θx")
#         axs[i,j].set_ylabel("θy")
#         axs[i,j].set_title("Dc={:8.5}".format(Dc[k]))
# plt.tight_layout()
# plt.suptitle("stacked and rebinned box slices")
# plt.savefig("stacked_and_rebinned_box_slices.png")
# plt.show()