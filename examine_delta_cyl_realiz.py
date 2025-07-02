import numpy as np
from matplotlib import pyplot as plt

allvals2=1e60*np.load("allvals2.npy") # indexed as [kpar,kperp,realization_number]
avg=np.mean(allvals2,axis=-1)
nrealiz=allvals2.shape[-1]
# maxunmod=allvals
# max0=
# max1=
# max2=

nplot=10
fig,axs=plt.subplots(nplot,nplot,figsize=(40,40))
for i in range(nplot):
    for j in range(nplot):
        idx=10*i+j
        im=axs[i,j].imshow(allvals2[:,:,idx])
        axs[i,j].set_xlabel("k$_{||}$ index")
        axs[i,j].set_ylabel("k$_\perp$ index")
        axs[i,j].set_title("realization "+str(idx))
        axs[i,j].set_aspect("equal")
        fig.colorbar(im,ax=axs[i,j])
im=axs[-1,-1].imshow(avg)
fig.colorbar(im,ax=axs[-1,-1])
axs[-1,-1].set_title("AVERAGE of "+str(nrealiz)+" realizations \n(if >100, some not imshown)")
plt.suptitle("further investigation of realizations of the delta-like response case")
plt.tight_layout()
plt.savefig("investig_mod2.png")
plt.show()