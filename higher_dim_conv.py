import numpy as np
from numpy.fft import rfft2,irfft2
from matplotlib import pyplot as plt
import time

def higher_dim_conv(f,g):
    if (f.shape!=g.shape):
        if(f.shape!=(g.T).shape):
            assert(1==0), "incompatible array shapes"
        else: # need to transpose g for things to work as expected
            g=g.T
    a,b=f.shape # if eval makes it to this point, f.shape==g.shape is True

    fp=np.zeros((2*a,2*b)) # holders for padded versions (wraparound-safe...)
    gp=np.zeros((2*a,2*b))
    fp[:a,:b]=f # populate the zero-padded versions
    gp[:a,:b]=g

    Fp=rfft2(fp)
    Gp=rfft2(gp)
    Fourier_space_product_p=Fp*Gp # _p for padded
    result_p=irfft2(Fourier_space_product_p)
    result=result_p[:a,:b]
    return result

A, B = 1010, 327
Avec=np.arange(A)
Bvec=np.arange(B)
P=np.outer(np.exp(-(Avec/(0.2*A))**2),np.exp(-(Bvec/(0.2*B))**2))
Barr,Aarr=np.meshgrid(Bvec,Avec)
W=np.eye(A,M=B)
t0=time.time()
windowed=higher_dim_conv(W,P)
t1=time.time()
print("convolution-based windowing took {:6.4} s".format(t1-t0))

case="no_zero_padding"
case="wraparound_safe"
savename="check_shortcut_"+case+".png"
fig,axs=plt.subplots(2,2)
im=axs[0,0].imshow(W,       extent=[0,B,A,0]) #,aspect=B/A)
plt.colorbar(im,ax=axs[0,0])
axs[0,0].set_title("W[$\Delta$ k$_{||}$,$\Delta$ k$_\perp$]")
im=axs[0,1].imshow(P,    extent=[0,B,A,0]) #,aspect=B/A)
plt.colorbar(im,ax=axs[0,1])
axs[0,1].set_title("P[k'$_{||}$,k'$_\perp$]")
im=axs[1,0].imshow(windowed, extent=[0,B,A,0]) #,aspect=B/A)
plt.colorbar(im,ax=axs[1,0])
axs[1,0].set_title("Pcont[k$_{||}$,k$_\perp$]")
plt.tight_layout()
plt.savefig(savename)
plt.show()