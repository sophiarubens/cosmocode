import numpy as np
from matplotlib import pyplot as plt

def total_baselines(alpha,beta):
    N=alpha*beta
    return N*(N-1)/2

def unique_baselines(alpha,beta):
    return 2*alpha*beta-alpha-beta

# alpha_sp=8.5
# beta_sp=6.3
# baseline_redundancy=np.genfromtxt("baseline_redundancy.csv",delimiter=",")
# print("baseline_redundancy.shape=",baseline_redundancy.shape)
# baseline_lengths=alpha_sp*baseline_redundancy[:,1]+beta_sp*(baseline_redundancy[:,2]+baseline_redundancy[:,3])
# plt.figure()
# plt.hist(baseline_lengths,bins=100)
# plt.xlabel("baseline length (m)")
# plt.ylabel("number of unique baselines")
# plt.title("hole-free CHORD baseline length distribution")
# plt.savefig("hole_free_CHORD_bl_len_distrib.png")
# plt.show()
# assert(1==0), "focusing on baseline length distribution"

# alpha=24
# beta=22
# Nant=alpha*beta
# total_bl_stats=np.zeros(Nant*(Nant-1)/2)
# for i in range(alpha):
#     vert_lim=np.max([i,alpha-i]) # only consider valid baselines (i.e. not off the array)
#     for j in range(i,beta): # only consider half of the possible reference points in the array to avoid double-counting while making sure to cover everything
#         horiz_lim=np.max([j,beta-j])
#         for n_pix_up_down in range(vert_lim):
#             for n_pix_right in range(horiz_lim):
#                 for n_pix_left in range(horiz_lim):
                    
                    

assert(1==0), "repeating for total instead of unique baselines"
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