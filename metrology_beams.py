import numpy as np
from matplotlib import pyplot as plt
from numpy.fft import fftn,ifftn,fftshift,ifftshift

d1x, d1y, d1z, d1p= np.loadtxt("CHORD_dish_1.txt").T
# d1gx,d1gy,d1gz,d1gp=np.loadtxt("CHORD_dish_1_gridded.txt").T

# print("d1x.shape,d1y.shape,d1z.shape,d1p.shape=",d1x.shape,d1y.shape,d1z.shape,d1p.shape) # unpacked as expected/ looks okay
# print("d1gx.shape,d1gy.shape,d1gz.shape,d1gp.shape=",d1gx.shape,d1gy.shape,d1gz.shape,d1gp.shape)

nbins=256
half=nbins//2
hist_fidu,edges= np.histogramdd(np.array((d1x,d1y,d1z)).T,     bins=nbins)
hist_pert,_=     np.histogramdd(np.array((d1x,d1y,d1z-d1p)).T, bins=edges)
x,y,z=edges

# try taking slices to plot??
slice_fidu_x=hist_fidu[half,:,:]
slice_fidu_y=hist_fidu[:,half,:]
slice_fidu_z=hist_fidu[:,:,half]
fig,axs=plt.subplots(1,3,figsize=(12,5))
axs[0].imshow(slice_fidu_x,origin="lower")
axs[1].imshow(slice_fidu_y,origin="lower")
axs[2].imshow(slice_fidu_z,origin="lower")
plt.savefig("examine_squared_fts.png")
plt.show()

xx,yy,zz=np.meshgrid(x,y,z,indexing="ij")

ft_fidu=fftn(ifftshift(hist_fidu))
beam_fidu=fftshift((ft_fidu*np.conjugate(ft_fidu)).real) # still stored as complex numbers, but the math says this quantity should be real and things are near the double precision floor, so I'm not worried about throwing away the imaginary part

ft_pert=fftn(ifftshift(hist_pert))
beam_pert=fftshift((ft_pert*np.conjugate(ft_pert)).real)
beam_diff=beam_fidu-beam_pert # ffts are linear, but the squaring makes the beam calc nonlinear, so this is different from the option below

ft_resi=fftn(ifftshift(hist_fidu-hist_pert))
beam_resi=fftshift((ft_resi*np.conjugate(ft_resi)).real)

# try taking slices to plot??
slice_fidu_x=beam_fidu[half,:,:]
slice_fidu_y=beam_fidu[:,half,:]
slice_fidu_z=beam_fidu[:,:,half]
fig,axs=plt.subplots(1,3,figsize=(12,5))
axs[0].imshow(slice_fidu_x,origin="lower")
axs[1].imshow(slice_fidu_y,origin="lower")
axs[2].imshow(slice_fidu_z,origin="lower")
plt.savefig("examine_squared_fts.png")
plt.show()