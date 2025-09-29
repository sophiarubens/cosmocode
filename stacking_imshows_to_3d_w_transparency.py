import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import Blues
from cosmo_distances import *

# import images from box
print("about to import box")
images=np.load("pert_box.npy")
print("imported box")

nu_ctr=400.
bw=nu_ctr/15. # reverse-engineering the class calculation for the call used when creating that box
N_chan=int(bw/0.183)
nu_lo=nu_ctr-bw/2.
nu_hi=nu_ctr+bw/2.
surv_channels=np.linspace(nu_hi,nu_lo,N_chan)
Dc=np.array([comoving_distance(ch) for ch in surv_channels])
print("N_chan=",N_chan)

pct2=np.percentile(images,2)
pct98=np.percentile(images,98)
cases=[0,-1]

fig,axs=plt.subplots(1,2,figsize=(15,5))
for i,case in enumerate(cases):
    axs[case].imshow(images[case], origin="lower", vmin=pct2,vmax=pct98, cmap=Blues)
    axs[i].set_xlabel("θx")
    axs[i].set_ylabel("θy")
    axs[case].set_title("Dc={:8.5}".format(Dc[case]))
plt.suptitle("verifying image stacking scaling intuition—400 MHz survey")
plt.savefig("verifying_stacking_scaling.png")
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