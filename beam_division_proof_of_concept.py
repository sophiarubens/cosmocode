import numpy as np
from matplotlib import pyplot as plt
from power import *

"""Helper variables"""

pstart='/Users/sophiarubens/Downloads/research/code/power_spectrum_dev/sample_boxes_21cmfast/a_box_z_'
zlate=5
zearly=10
nz=zearly-zlate+1
allzs=np.arange(zearly,zlate-1,-1)
strzs=[str(n) for n in range(zearly,zlate-1,-1)]
# allzs, strzs, boxes: TIME ADVANCES / REDSHIFT DECREASES AS YOU MOVE FORWARD THROUGH THE ARRAY

"""Take a "pristine" box of simulated cosmological 21-cm brightness temperature"""

fig,axs=plt.subplots(1,6,figsize=(15,2.2))
slicee=50
for i in range(nz):
    bwd=nz-i-1
    box = np.load(pstart+strzs[i]+'.npy') # import box
    if (i==0): # must figure out how big to make the box holder
        box0,box1,box2 = box.shape
        boxes = np.zeros((nz,box0,box1,box2)) # 4D structure to hold boxes ... treat as an array of 3D arrays
        xHs = np.zeros(nz)
    xHs[i] = np.count_nonzero(box)/(box0*box1*box2)
    boxes[i,:,:,:] = box # store each box in the holder ... will come in handy later
    im=axs[i].imshow(box[:,:,slicee]) # plot a colo[u]r map of a slice in the middle
    fig.colorbar(im)
    axs[bwd].set_xlabel('voxel index')
    axs[bwd].set_ylabel('voxel index')
    axs[bwd].set_title('z='+strzs[bwd]+'; x̄H='+str(round(xHs[i],3)))
fig.suptitle('21-cm brightness temperature in pristine 21cmFast boxes')
fig.tight_layout()
plt.show()

"""Construct the "pristine" power spectra"""

twopi2=2*np.pi**2
Lsurv=100
nbins=12
powers=np.zeros((nz,nbins))

plt.figure(figsize=(10,4))
for i in range(nz):
    bwd=nz-i-1
    kfloors,vals = ps_autobin(boxes[i,:,:,:],'log',Lsurv,nbins) # since the boxes and PS calls are the same, it doesn't matter that we overwrite kfloors each time, b/c the kfloors are the same at each redshift in this loop
    powers[bwd,:] = vals
    plt.loglog(kfloors,kfloors**3*vals/twopi2,label='z='+strzs[i]+'; x̄H='+str(round(xHs[bwd],3)),marker='.')

plt.xlabel('$h^{-1}k$')
plt.ylabel('$\Delta^2$')
plt.title('$\Delta^2$ for various 21cmFast simulation boxes')
plt.legend()
plt.tight_layout()
plt.show()

np.savetxt('z5spec.txt',np.asarray([kfloors,vals]).T)

"""Multiply the box data by a fake Gaussian [primary] beam in configuration space"""

lim=10
vec=np.linspace(-lim,lim,box0) # artificial coordinate system w/ no physical basis useful only for making pts at which to evaluate a 3D Gaussian ... slightly hacky b/c I'm assuming for the first time that the box is cubic
X,Y,Z=np.meshgrid(vec,vec,vec)
sig=0.8*lim
mu=0
pre=1/(sig**3*(2*np.pi)**1.5)
gau3=np.exp(-(X**2+Y**2+Z**2)/(2*sig**2))
beam=pre*gau3

beamed=np.zeros((nz,box0,box1,box2))
fig,axs=plt.subplots(1,6,figsize=(15,2.2))
for i in range(nz):
    bwd=nz-i-1
    beamed[i,:,:,:]=beam*boxes[i,:,:,:] # the important step
    im=axs[i].imshow(box[:,:,slicee])
    fig.colorbar(im)
    axs[bwd].set_xlabel('voxel index')
    axs[bwd].set_ylabel('voxel index')
    axs[bwd].set_title('z='+strzs[bwd]+'; x̄H='+str(round(xHs[i],3)))
fig.suptitle('21-cm brightness temperature in beam-modulated 21cmFast boxes')
fig.tight_layout()
plt.show()

"""FT -> attenuate -> add random noise -> IFT -> try to divide out primary beam"""

BEAMED   = np.zeros((nz,box0,box1,box2),dtype='complex128') # FT(box*beam) for each box
ATTD     = np.zeros((nz,box0,box1,box2),dtype='complex128') # ^, attentuated in Fourier space
NOISY    = np.zeros((nz,box0,box1,box2),dtype='complex128') # ^, infused with noise in Fourier space
config   = np.zeros((nz,box0,box1,box2))   # ^, IFTd back into configuration space
unbeamed = np.zeros((nz,box0,box1,box2))   # ^, with the beam divided out (screw up large scales on purpose)

prop=0.5e-2 # fractional amplitude of the Gaussian from which the noise is drawn, relative to the max of the attenuated, beamed, FTd box

for i in range(nz):
    bwd=nz-i-1
    BEAMED[i,:,:,:]   = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(beamed[i,:,:,:])))      # FT
    ATTD[i,:,:,:]     = gau3*BEAMED[i,:,:,:]                                                 # apply attenuation
    noise             = prop*np.max(ATTD[i,:,:,:])*np.random.randn(box0,box1,box2)           # construct noise
    NOISY[i,:,:,:]    = ATTD[i,:,:,:]+noise                                                  # apply noise
    config[i,:,:,:]   = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(NOISY[i,:,:,:]))).real # IFT (.imag=0)       #     print(np.mean(test),np.std(test))
    unbeamed[i,:,:,:] = config[i,:,:,:]/beam                                                 # divide out beam

"""Construct the "warped at high k" power spectra"""

recovered=np.zeros((nz,nbins))
colors=['C0','C1','C2','C3','C4','C5']

plt.figure(figsize=(10,4))
for i in range(nz):
    bwd=nz-i-1
    kfloors,vals = ps_autobin(unbeamed[i,:,:,:],'log',Lsurv,nbins) # once again, ok to overwrite kfloors
    recovered[bwd,:] = vals
    plt.loglog(kfloors,kfloors**3*powers[bwd,:]/twopi2,color=colors[i],marker='.',label='z='+strzs[i]+'; x̄H='+str(round(xHs[bwd],3)))
    plt.loglog(kfloors,kfloors**3*vals/twopi2,color=colors[i],marker='.',linestyle='--')

plt.xlabel('$h^{-1}k$')
plt.ylabel('$\Delta^2$')
plt.title('Dimensionless PS for various beam-misprocessed (dashed) and pristine (solid) 21cmFast simulation boxes')
plt.legend()
plt.tight_layout()
plt.show()

"""problem before: I was adding noise far smaller than the attenuation, so somehow the attenuation won out and the power was being suppressed ... this isn't physically realistic

Things to look into:
* how can you make a smart choice of how much attenuation versus noise to apply?
"""