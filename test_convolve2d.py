from scipy.signal import convolve2d,convolve
from numpy.fft import rfft2,irfft2,fft2,ifft2
import scipy
import numpy as np
from matplotlib import pyplot as plt

def get_padding(n):
    padding=n-1
    padding_lo=int(np.ceil(padding / 2))
    padding_hi=padding-padding_lo
    return np.array((padding_lo,padding_hi))

Wcont=np.load("cyl_Wcont.npy")
P=np.load("cyl_P.npy")
s0,s1=Wcont.shape

ptop,pbot=get_padding(s0)
plhs,prhs=get_padding(s1)
# ptop=s0
# pbot=s0
# plhs=s1
# prhs=s1

def centered(arr, newshape):
    # Return the center newshape portion of the array.
    newshape = np.asarray(newshape)
    currshape = np.array(arr.shape)
    startind = (currshape - newshape) // 2
    endind = startind + newshape
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    print("myslice=",myslice)
    return arr[tuple(myslice)],myslice

Wcontp=np.pad(Wcont,((ptop, pbot), (plhs, prhs)),
                    # mode="constant", constant_values=0)
                    mode="symmetric", reflect_type="even")

# ##
# s0,s1=Pshape # by now, P and Wcont have the same shapes

Pcont_plain=convolve2d(P,Wcont)
peak0,peak1=np.unravel_index(np.argmax(Pcont_plain, axis=None), Pcont_plain.shape)
# print("peak0,peak1=",peak0,peak1)
Pcont_sliced=Pcont_plain[peak0:peak0+s0:,peak1:peak1+s1]
# print("Pcont_sliced.shape=",Pcont_sliced.shape)
# ##

Pcont=convolve2d(P,Wcont)
peak0,peak1=np.unravel_index(np.argmax(Pcont, axis=None), Pcont.shape)
# print("peak0,peak1=",peak0,peak1)
# Pcont_sliced=Pcont[peak0:peak0+s0:,peak1:peak1+s1]

Pcont_valid,myslice=centered(Pcont,P.shape) 

Pcont_ifftshifted=np.fft.ifftshift(Pcont)

print("Wcont.shape=",Wcont.shape)
print("P.shape=",P.shape)
print("Pcont.shape=",Pcont.shape)

fig,axs=plt.subplots(1,6,figsize=(15,5))

im=axs[0].pcolor(P)
plt.colorbar(im,ax=axs[0])
axs[0].set_title("P pristine")
im=axs[1].pcolor(Wcontp)
plt.colorbar(im,ax=axs[1])
axs[1].set_title("Wcont-padded")
im=axs[2].pcolor(Wcont)
plt.colorbar(im,ax=axs[2])
axs[2].set_title("Wcont (actually Wtrue atm)")
im=axs[3].pcolor(Pcont)
plt.colorbar(im,ax=axs[3])
myslice0=myslice[0]
myslice1=myslice[1]
print("myslice0,myslice1=",myslice0,myslice1)
axs[3].axhline(myslice0.start)
axs[3].axhline(myslice0.stop)
axs[3].axvline(myslice1.start)
axs[3].axvline(myslice1.stop)
axs[3].set_title("Pcont (actually Ptrue atm)")
im=axs[4].pcolor(Pcont_valid)
plt.colorbar(im,ax=axs[4])
axs[4].set_title("Pcont_valid (actually Ptrue_valid atm)")
im=axs[5].pcolor(Pcont_sliced)
plt.colorbar(im,ax=axs[5])
axs[5].set_title("Pcont-sliced")
for i in range(2):
    # axs[i].axhline(perp_line, label="perp sigma for analyt Wtrue", c="C0")
    # axs[i].axvline(par_line,  label="par sigma for analyt Wtrue",  c="C1")
    # if i==0:
        # print("adding par_line and perp_line to plot 0:",par_line,perp_line)

    axs[i].set_xlabel("k$_{||}$ (Mpc$^{-1}$)")
    axs[i].set_ylabel("k$_\perp$ (Mpc$^{-1}$)")

    # axs[i].legend()
plt.suptitle("inspect where the analytical shape issue enters")
plt.tight_layout()
plt.savefig("test_with_convolve2d.png")
plt.show()