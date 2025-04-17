import numpy as np
from matplotlib import pyplot as plt

def total_baselines(alpha,beta):
    N=alpha*beta
    return N*(N-1)/2

def unique_baselines(alpha,beta):
    return 2*alpha*beta-alpha-beta

vec=np.arange(1,25)
alpha,beta=np.meshgrid(vec,vec)

fig,axs=plt.subplots(1,3,figsize=(20,5))
total=total_baselines(alpha,beta)
im=axs[0].imshow(total,extent=[vec[0],vec[-1],vec[-1],vec[0]]) # lrbt
axs[0].set_xlabel("alpha = number of antenna rows")
axs[0].set_ylabel("beta = number of antenna columns")
fig.colorbar(im,ax=axs[0])
axs[0].set_title("total")
unique=unique_baselines(alpha,beta)
im=axs[1].imshow(unique,extent=[vec[0],vec[-1],vec[-1],vec[0]])
axs[1].set_xlabel("alpha = number of antenna rows")
axs[1].set_ylabel("beta = number of antenna columns")
fig.colorbar(im,ax=axs[1])
axs[1].set_title("unique")
dif=total-unique
im=axs[2].imshow(dif,extent=[vec[0],vec[-1],vec[-1],vec[0]])
axs[2].set_xlabel("alpha = number of antenna rows")
axs[2].set_ylabel("beta = number of antenna columns")
fig.colorbar(im,ax=axs[2])
axs[2].set_title("total-unique")
plt.suptitle("number of baselines for rectangular alpha-by-beta interferometer arrays")
plt.savefig("rect_array_baseline_stats.png")
plt.show()