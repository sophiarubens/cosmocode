import numpy as np
from matplotlib import pyplot as plt
import time

lim=25
npix=128
nbl=278256
nhr=15
x0=np.random.randint(-1.5*lim,1.5*lim,(nbl,nhr))
y0=np.random.randint(-1.5*lim,1.5*lim,(nbl,nhr))
sigmas=np.abs(np.ones((nbl,nhr))+np.random.randn(nbl,nhr))
vec=np.linspace(-lim,lim,npix)
xx,yy=np.meshgrid(vec,vec,indexing="ij")

def template(x,y,x0,y0,sigma=1):
    return np.exp(-((x-x0)**2+(y-y0)**2)/(2*sigma**2) )

smeared=np.zeros((npix,npix))
t0=time.time()
for i in range(nbl):
    for j in range(nhr):
        delta=template(xx,yy,x0[i,j],y0[i,j],sigma=sigmas[i,j])
        smeared+=delta
t1=time.time()
print("added",nbl,"smear contributions to the uv-plane in",t1-t0,"s")

plt.figure()
plt.imshow(smeared,origin="lower",extent=[vec[0],vec[-1],vec[0],vec[-1]])
plt.savefig("smeared_mock_uv_"+str(npix)+"_pix_"+str(nbl)+"_bl_"+str(nhr)+"_hr_"+str(int(t1-t0))+"_s.png")
plt.show()