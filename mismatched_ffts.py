import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import CenteredNorm
from scipy.fft import fftn,irfftn, fftshift,ifftshift

rng=np.random.default_rng()
sz=(10,11)
a=rng.normal(size=sz)
A=fftshift(fftn(ifftshift(a)))
a_E2E=fftshift(irfftn(ifftshift(A),s=sz))
fig,axs=plt.subplots(1,5,layout="constrained")
ar=0.5*np.abs(np.max((a,a_E2E)))
Ar=0.5*np.abs(np.max((A)))
im=axs[0].imshow(a.T,origin="lower",norm=CenteredNorm(vcenter=0,halfrange=ar))
axs[0].set_title("a")
plt.colorbar(im,ax=axs[0])
im=axs[1].imshow(A.real.T,origin="lower",norm=CenteredNorm(vcenter=0,halfrange=Ar))
axs[1].set_title("Re[A]")
plt.colorbar(im,ax=axs[1])
im=axs[2].imshow(A.imag.T,origin="lower",norm=CenteredNorm(vcenter=0,halfrange=Ar))
axs[2].set_title("Im[A]")
plt.colorbar(im,ax=axs[2])
im=axs[3].imshow(a_E2E.T,origin="lower",norm=CenteredNorm(vcenter=0,halfrange=ar))
axs[3].set_title("a E2E")
plt.colorbar(im,ax=axs[3])
im=axs[4].imshow((a-a_E2E).T,origin="lower",norm=CenteredNorm(vcenter=0))
axs[4].set_title("a - a E2E")
plt.colorbar(im,ax=axs[4])
plt.savefig("mismatched_ffts.png")