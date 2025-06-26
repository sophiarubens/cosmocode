import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import convolve
from bias_helper_fcns import W_cyl_binned

def get_padding(n):
    padding=n-1
    padding_lo=int(np.ceil(padding / 2))
    padding_hi=padding-padding_lo
    return padding_lo,padding_hi

def higher_dim_conv(P,Wcont):
    Pshape=P.shape
    Wcontshape=Wcont.shape
    if (Pshape!=Wcontshape):
        if(Pshape.T!=Wcontshape):
            assert(1==0), "window and pspec shapes must match"
        Wcont=Wcont.T # force P and Wcont to have the same shapes
    s0,s1=Pshape # by now, P and Wcont have the same shapes
    pad0lo,pad0hi=get_padding(s0)
    pad1lo,pad1hi=get_padding(s1)

    # v1 - bad
    # Pp=np.pad(P,((pad0lo,pad0hi),(pad1lo,pad1hi)),"constant",constant_values=((0,0),(0,0)))
    # conv=convolve(Wcont,Pp,mode="valid") 

    # v2 - good
    Wcontp=np.pad(Wcont,((pad0lo,pad0hi),(pad1lo,pad1hi)),"edge")
    conv=convolve(Wcontp,P,mode="valid")

    # # v3 - bad
    # Wcontp=np.pad(Wcont,((pad0lo,pad0hi),(pad1lo,pad1hi)),"constant",constant_values=((0,0),(0,0)))
    # conv=convolve(Wcontp,P,mode="valid")
    return conv

kparvec=np.linspace(0.019,0.979,52)
kperpvec=np.linspace(0.0849,1.29,123)
r0=2210
kpargrid,kperpgrid=np.meshgrid(kparvec,kperpvec,indexing="ij")
modulate_width=0.05
P=np.exp(-(kpargrid/(2*modulate_width))**2-(kperpgrid/(2*modulate_width))**2)
W_narrow=W_cyl_binned(kparvec,kperpvec,4.1,r0,1.4e-3) # realistic-ish values
W_broad=W_cyl_binned(kparvec,kperpvec,1e-10,r0,1e-10) # W_cyl_binned(kpar,kperp,sigLoS,r0,fwhmbeam,save=False)
W_broad_2=W_cyl_binned(kparvec,kperpvec,4.1e-7,r0,1.4e-10)
P_conv_W_narrow=higher_dim_conv(P,W_narrow)
P_conv_W_broad= higher_dim_conv(P,W_broad)
# P_conv_W_broad= higher_dim_conv(P,W_broad_2 )

# print("P.shape,W_narrow.shape,W_broad.shape,P_conv_W_narrow.shape,P_conv_W_broad.shape=",P.shape,W_narrow.shape,W_broad.shape,P_conv_W_narrow.shape,P_conv_W_broad.shape)

fig,axs=plt.subplots(2,3,figsize=(15,5))
im=axs[0,0].pcolor(kpargrid,kperpgrid,P)
cbar=plt.colorbar(im,ax=axs[0,0],extend="both")
axs[0,0].set_title("P")
im=axs[1,0].pcolor(kpargrid,kperpgrid,P)
cbar=plt.colorbar(im,ax=axs[1,0],extend="both")
axs[1,0].set_title("P")
im=axs[0,1].pcolor(kpargrid,kperpgrid,W_narrow)
cbar=plt.colorbar(im,ax=axs[0,1],extend="both")
axs[0,1].set_title("W_narrow")
im=axs[1,1].pcolor(kpargrid,kperpgrid,W_broad)
cbar=plt.colorbar(im,ax=axs[1,1],extend="both")
axs[1,1].set_title("W_broad")
im=axs[0,2].pcolor(kpargrid,kperpgrid,P_conv_W_narrow)
cbar=plt.colorbar(im,ax=axs[0,2],extend="both")
axs[0,2].set_title("P_conv_W_narrow")
im=axs[1,2].pcolor(kpargrid,kperpgrid,P_conv_W_broad)
cbar=plt.colorbar(im,ax=axs[1,2],extend="both")
axs[1,2].set_title("P_conv_W_broad")
for i in range(2):
    for j in range(3):
        axs[i,j].set_xlabel("k$_{||}$")
        axs[i,j].set_ylabel("k$_\perp$")
plt.suptitle("generalize the convolution padding to work for broad AND narrow windows")
plt.tight_layout()
plt.savefig("broad_narrow_conv_test.png")
plt.show()