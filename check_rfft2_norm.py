import numpy as np
from numpy.fft import rfft2, irfft2
from scipy.signal import convolve

a=np.array([[1,0,-9],[0, 1,0]])
ashape0,ashape1=a.shape
b=np.array([[1,2,4],[6,-1,3]])
zblock=np.zeros((2,3))
zblock4=np.zeros((4,6))
azp=np.vstack((np.hstack((a,zblock)),np.hstack((zblock,zblock))))
bzp=np.vstack((np.hstack((b,zblock)),np.hstack((zblock,zblock))))
azp3=np.stack((azp,zblock4,zblock4),axis=2)
bzp3=np.stack((bzp,zblock4,zblock4),axis=2)

A=rfft2(azp)
B=rfft2(bzp)
AB=A*B
aconvb=irfft2(AB)
aconvb_scipy=convolve(a,b)
# aconvb_scipy=convolve(azp3,bzp3,mode="valid")

print("a=\n",a,"\n")
print("b=\n",b,"\n")
# print("A=\n",A,"\n")
# print("B=\n",B,"\n")
# print("AB=\n",AB,"\n")
print("aconvb=\n",aconvb,"\n")
print("aconvb_scipy=\n",aconvb_scipy,"\n")
# print("aconvb_scipy[:ashape0,:ashape1]=\n",aconvb_scipy[:ashape0,:ashape1],"\n")