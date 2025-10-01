import numpy as np
from matplotlib import pyplot as plt
from numpy.fft import ifft2,fftshift,ifftshift
from matplotlib.cm import Blues

lim=1e6
Npix=1024
uv=np.linspace(-lim,lim,Npix)
du=uv[1]-uv[0]
d2u=du**2
uu,vv=np.meshgrid(uv,uv,indexing="ij")

stripes=0.*uu
Nstripes=24
spacing=int(Npix//Nstripes)
for j in range(Nstripes):
    lo=j*spacing
    stripes[:,lo:lo+4]=1.
iftstripes=np.abs(fftshift(ifft2(ifftshift(stripes)*d2u,norm="forward")))

bite_mask=np.sqrt(uu**2+vv**2)
bite_mask[np.logical_and((bite_mask<lim), (np.abs(uu)<0.5*lim))]=False
bite_mask[~np.logical_and((bite_mask<lim), (np.abs(uu)<0.5*lim))]=True
# plt.figure()
# plt.imshow(bite_mask)
# plt.show()
bite_mask=np.asarray(bite_mask,dtype=bool)
print("bite_mask.shape=",bite_mask.shape)
bite=np.copy(stripes)
print("bite.shape=",bite.shape)
bite=np.ma.masked_where(~bite_mask,bite)
# plt.figure()
# plt.imshow(bite)
# plt.show()
iftbite=np.abs(fftshift(ifft2(ifftshift(bite)*d2u,norm="forward")))

fig,axs=plt.subplots(2,2,figsize=(12,8))
im=axs[0,0].imshow(stripes,origin="lower",cmap=Blues)
axs[0,0].set_title("stripes")
plt.colorbar(im,ax=axs[0,0])
lo=15
im=axs[0,1].imshow(iftstripes,origin="lower",cmap=Blues,vmin=np.percentile(iftstripes,lo),vmax=np.percentile(iftstripes,100-lo))
axs[0,1].set_title("IFT(stripes)")
plt.colorbar(im,ax=axs[0,1])
im=axs[1,0].imshow(bite,origin="lower",cmap=Blues)
axs[1,0].set_title("bite")
plt.colorbar(im,ax=axs[1,0])
im=axs[1,1].imshow(iftbite,origin="lower",cmap=Blues,vmin=np.percentile(iftbite,lo),vmax=np.percentile(iftbite,100-lo))
axs[1,1].set_title("IFT(bite)")
plt.colorbar(im,ax=axs[1,1])
plt.tight_layout()
plt.savefig("phenomena.png")
plt.show()