import numpy as np
from matplotlib import pyplot as plt
from bias_helper_fcns import higher_dim_conv
from scipy.signal import convolve

###
def same_conv(x, k):
    if len(k) > len(x):
        # consider longer as x and other as kernel
        x, k = k, x

    n = x.shape[0]
    m = k.shape[0]

    padding   = m - 1
    left_pad  = int(np.ceil(padding / 2))
    right_pad = padding - left_pad

    x = np.pad(x, (left_pad, right_pad), 'constant')
    # print(len(x))

    out = []

    # flip the kernel
    k = k[::-1]
    # print(k)
    
    for i in range(n):
        out.append(np.dot(x[i: i+m], k))

    return np.array(out)
###

def get_padding(n):
    padding=n-1
    padding_lo=int(np.ceil(padding / 2))
    padding_hi=padding-padding_lo
    return padding_lo,padding_hi

def same_conv_2d(P,Wcont):
    Pshape=P.shape
    Wcontshape=Wcont.shape
    if (Pshape!=Wcontshape):
        if(Pshape.T!=Wcontshape):
            assert(1==0), "window and pspec shapes must match"
        Wcont=Wcont.T # force P and Wcont to have the same shapes
    # by now, P and Wcont have the same shapes
    s0,s1=Pshape
    pad0lo,pad0hi=get_padding(s0)
    pad1lo,pad1hi=get_padding(s1)
    Pp=np.pad(P,((pad0lo,pad0hi),(pad1lo,pad1hi)),"constant",constant_values=((0,0),(0,0)))
    conv=convolve(Wcont,-Pp,mode="valid")
    # print("pad0lo,pad0hi,s0,  pad1lo,pad1hi,s1", pad0lo,pad0hi,s0,  pad1lo,pad1hi,s1)
    # plt.figure()
    # plt.imshow(conv)
    # plt.colorbar()
    # plt.title("check conv truncation")
    # plt.show()
    return conv

P=     np.load("cyl_P.npy")
Wcont= np.load("cyl_Wcont.npy")
kparg= np.load("cyl_kpargrid.npy")
kperpg=np.load("cyl_kperpgrid.npy")
nkpar,nkperp=P.shape

test1=higher_dim_conv(P,Wcont) 
# test2=(convolve(Wcont.T,P.T).T)[:nkpar,:nkperp] # without the flip, the peak feature is in the top left corner and there are more vertical pixels
test2=same_conv_2d(Wcont,P)
# test3=convolve(Wcont.T,P.T)
print("test1.shape,test2.shape=",test1.shape,test2.shape)

nplot=4
fig,axs=plt.subplots(1,nplot,figsize=(20,5))
axs[0].pcolor(kparg,kperpg,P)
# axs[0].imshow(P)
axs[0].set_title("P")
axs[1].pcolor(kparg,kperpg,Wcont)
# axs[1].imshow(Wcont)
axs[1].set_title("Wcont")
axs[2].pcolor(kparg,kperpg,test1)
# axs[2].imshow(test1)
axs[2].set_title("Pcont - legacy")
axs[3].pcolor(kparg,kperpg,test2)
# axs[3].imshow(test2)
axs[3].set_title("Pcont - new")

for i in range(nplot):
    axs[i].set_aspect("equal")
    axs[i].set_xlabel("$k_{||}$")
    axs[i].set_ylabel("$k_\perp$")
plt.suptitle("convolution scrutiny")
plt.tight_layout()
plt.savefig("scrutinize_convolution.png")
plt.show()