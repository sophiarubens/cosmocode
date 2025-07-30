import numpy as np
from matplotlib import pyplot as plt
from power import *
from bias_helper_fcns import *
from power_class import *
import time

import powerbox as pbox
from powerbox.powerbox import _magnitude_grid
from powerbox.dft import fft,ifft # normalization and Fourier dual characterized by (a,b)=(0,2pi) (the cosmology convention), although the numpy convention is (a,b)=(1,1)
# useful
pi=np.pi
twopi=2.*pi

# box and spec conditioning
# L = 126
# N = 512 # ~11 s/realiz
L=126
N=54
dx = L/N
num_modes=15
fac=99

############################## NON-DETERMINISTIC DIRECTION
Nrealiz=500
idx=-0.9 # -0.9 0. 2.3

T_pb_realizations_means=np.zeros(Nrealiz)
T_pb_realizations_stds= np.zeros(Nrealiz)
P_pb_summed=      np.zeros((N,N,N))
t0=time.time()
t1=time.time()
for i in range(Nrealiz):
    pb= pbox.PowerBox(N=N,dim=3,
                      pk = lambda k: k**idx, # fiducial power spec
                      boxlength = L          # determines units of k
                     )
    T_pb= pb.delta_x()
    T_pb_realizations_means[i]=np.mean(T_pb)
    T_pb_realizations_stds[i]= np.std( T_pb)
    P_pb=pb.power_array()
    P_pb_summed+=P_pb
T_pb_mean=np.mean(T_pb_realizations_means)
T_pb_std= np.mean(T_pb_realizations_stds)
P_pb=P_pb_summed/Nrealiz
print("T_pb_mean= {:6.4} \nT_pb_std= {:6.4}".format(T_pb_mean,T_pb_std))
nrow,ncol=3,6
vmi=np.min(P_pb)
vma=np.percentile(P_pb,fac)
fig,axs=plt.subplots(nrow,ncol,figsize=(20,9))
for i in range(nrow):
    for j in range(ncol):
        if i==0:
            im=axs[i,j].imshow(P_pb[j*N//ncol,:,:],vmin=vmi,vmax=vma)
            title="P_pb["+str(j*N//ncol)+",:,:]"
        elif i==1:
            im=axs[i,j].imshow(P_pb[:,j*N//ncol,:],vmin=vmi,vmax=vma)
            title="P_pb[:,"+str(j*N//ncol)+",:]"
        else:
            im=axs[i,j].imshow(P_pb[:,:,j*N//ncol],vmin=vmi,vmax=vma)
            title="P_pb[:,:,"+str(j*N//ncol)+"]"
        axs[i,j].set_title(title)
fig.colorbar(im)
plt.suptitle("selected slices of powerbox unbinned power spectra")
plt.savefig("slices_pb_unbinned_spectra.png")
plt.show()

# need to pick the k-modes at which I sample the fiducial power spec to match those that powerbox provides internally
# otherwise, it will be hard to separate this implementation difference from actual algorithm issues
# that's the thing: powerbox does not go in the box->spec direction; you provide a functional form and it computes box realizations.
T_cs_realizations_means=np.zeros(Nrealiz)
T_cs_realizations_stds= np.zeros(Nrealiz)
P_cs_summed=np.zeros((N,N,N))
# k_fid=np.linspace(twopi/L,pi*L/N)     # 50:  0.2133
# k_fid=np.linspace(twopi/L,pi*L/N,500) # 500: 0.5419
# k_fid=np.linspace(twopi/L,pi*N/L,N)   # 54:  0.4495 (same as the number of voxels per side at the time)
k_fid=np.linspace(twopi/L,pi*N/L,51)    # 51:  0.4397
cs= cosmo_stats(Lsurvey=L,P_fid=k_fid**idx,Nvox=N,Nk0=num_modes,realization_ceiling=Nrealiz,k_fid=k_fid)
for i in range(Nrealiz):
    cs.generate_box()
    T_cs_realizations_means[i]=np.mean(cs.T_pristine)
    T_cs_realizations_stds[i]= np.std(cs.T_pristine)
    cs.generate_P()
    P_cs_summed+=cs.unbinned_P
T_cs_mean=np.mean(T_cs_realizations_means)
T_cs_std= np.mean(T_cs_realizations_stds)
P_cs=P_cs_summed/Nrealiz
print("T_cs_mean= {:6.4} \nT_cs_std= {:6.4}".format(T_cs_mean,T_cs_std))
fig,axs=plt.subplots(nrow,ncol,figsize=(20,9))
vmi=np.min(P_cs)
vma=np.percentile(P_cs,fac)
for i in range(nrow):
    for j in range(ncol):
        if i==0:
            im=axs[i,j].imshow(P_cs[j*N//ncol,:,:],vmin=vmi,vmax=vma)
            title="P_cs["+str(j*N//ncol)+",:,:]"
        elif i==1:
            im=axs[i,j].imshow(P_cs[:,j*N//ncol,:],vmin=vmi,vmax=vma)
            title="P_cs[:,"+str(j*N//ncol)+",:]"
        else:
            im=axs[i,j].imshow(P_cs[:,:,j*N//ncol],vmin=vmi,vmax=vma)
            title="P_cs[:,:,"+str(j*N//ncol)+"]"
        axs[i,j].set_title(title)
fig.colorbar(im)
plt.suptitle("selected slices of cosmo_stats unbinned power spectra")
plt.savefig("slices_cs_unbinned_spectra.png")
plt.show()

# and check the pb version against the unbinned power spec I get from my code