import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import convolve

def get_padding(n):
    padding=n-1
    padding_lo=int(np.floor(padding / 2))
    padding_hi=padding-padding_lo
    return padding_lo,padding_hi

Wcont=np.load("Wcont.npy")
Pcyl=np.load( "Pcyl.npy" )
s0,s1=Pcyl.shape

pad0lo,pad0hi=get_padding(s0)
pad1lo,pad1hi=get_padding(s1)

Wcontp=np.pad(Wcont,((pad0lo,pad0hi),(pad1lo,pad1hi)),"edge")
# Wcontp=np.pad(Wcont,((pad0lo,pad0hi),(pad1lo,pad1hi)),"linear_ramp") # indistinguishable from ^
# Wcontp=np.pad(Wcont,((pad0lo,pad0hi),(pad1lo,pad1hi)),"constant") # indistinguishable from ^
# Wcontp=np.pad(Wcont,((pad1lo,pad1hi),(pad0lo,pad0hi)),"edge") # shape completely wrong, as expected
# Wcontp=np.pad(Wcont,((s0,s0),(s1,s1)),"edge")
# Pcont=convolve(Wcontp,Pcyl,mode="valid")
Pcont=convolve(Wcontp,Pcyl,mode="full")
print("Pcont.shape=",Pcont.shape)
peak0,peak1=np.unravel_index(Pcont.argmax(), Pcont.shape)
print("peak0,peak1=",peak0,peak1)
Pcont=Pcont[peak0:peak0+s0,peak1:peak1+s1]
print("Pcont.shape=",Pcont.shape)

a=7
fig,axs=plt.subplots(1,3,figsize=(10,5))
axs[0].imshow(Pcyl,origin="lower",aspect=a)
axs[0].set_title("Pcyl")
axs[1].imshow(Wcont,origin="lower",aspect=a)
axs[1].set_title("Wcont")
axs[2].imshow(Pcont,origin="lower",aspect=a)
axs[2].set_title("Pcont")
plt.savefig("pinpointed_convolution_tests.png")
plt.show()