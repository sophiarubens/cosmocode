import numpy as np
from matplotlib import pyplot as plt
from scipy.fft import fft,fftshift,ifftshift,fftfreq

indices=[0., -0.1, -1., -2.3, -9]
N_indices=len(indices)
N_points=1048
x=np.linspace(-0.49,0.5,N_points)
k=fftshift(fftfreq(N_points))
fig,axs=plt.subplots(2,N_indices, layout="constrained")
for i,idx in enumerate(indices):
    f=np.abs(x)**idx
    f/=f.max()
    f-=np.mean(f)
    F=fftshift(fft(ifftshift(f)))
    axs[0,i].plot(x,f)
    axs[1,i].plot(k,np.abs(F)**2)
    axs[0,i].set_xlabel("x")
    axs[1,i].set_xlabel("k")
    axs[0,i].set_ylabel("config amplitude")
    axs[1,i].set_ylabel("unnormalized power")
    if i==0:
        config_head="config space (field)\nidx="
        Fourier_head="Fourier space (power)\nidx="
    else:
        config_head=""
        Fourier_head=""
    stridx=str(idx)
    axs[0,i].set_title(config_head+stridx)
    axs[1,i].set_title(Fourier_head+stridx)
    axs[1,i].set_ylim(0,10)
plt.suptitle("How much does a power law need to decay over the domain\n before you avoid the numerical delta issue?")
plt.savefig("artificial_foregrounds_idx.png")

extents=[5e-3, 0.05, 0.5, 10, 1e3]
N_extents=len(extents)
fig,axs=plt.subplots(2,N_extents,layout="constrained")
for i,extent in enumerate(extents):
    x=np.linspace(1e-3,extent, N_points)
    f=x**-0.8
    f-=np.mean(f)
    F=fftshift(fft(ifftshift(f)))
    axs[0,i].plot(x,f)
    axs[1,i].plot(k,np.abs(F)**2)
    axs[0,i].set_xlabel("x")
    axs[1,i].set_xlabel("k")
    axs[0,i].set_ylabel("config amplitude")
    axs[1,i].set_ylabel("unnormalized power")
    if i==0:
        config_head="config space (field)\nidx="
        Fourier_head="Fourier space (power)\nidx="
    else:
        config_head=""
        Fourier_head=""
    strext=str(extent)
    axs[0,i].set_title(config_head+strext)
    axs[1,i].set_title(Fourier_head+strext)
    # axs[1,i].set_ylim(0,10)
plt.suptitle("How much domain do you need to sample a given power law\n over before you avoid the numerical delta issue?")
plt.savefig("artificial_foregrounds_ext.png")