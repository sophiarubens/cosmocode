import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm

# kvec=np.arange(0.1,0.6,0.1)
# k,kpr=np.meshgrid(kvec,kvec)
# print('k=\n',k,'kp=\n',kpr)

def Wmat(kvec,sigma): 
   k,kp=np.meshgrid(kvec,kvec)
   return (2*np.pi*sigma**2)**3*np.exp(-sigma**2*(k-kp)**2/2.)

npts=100

# Pobs(k) = W(k,kpr)Ptrue(kpr)
# ultimately interested in Pobs(k)
# start by doing something similar to what Hannah did in the ApJ paper:
# plot Ptrue(kpr) with a shared-axis plot below showing W(k,kpr)
# k=np.linspace(0.05,0.7,npts)
k=np.linspace(-50,50,npts)
amp=1529

sig=8
Ptrue1=amp*np.exp(-k**2/sig**2) # gaussian
Ptrue2=amp*np.ones(npts) # flat
Ptrue3=amp*k*np.exp(-k**2/sig**2) # skewed
Ptrue4=amp*np.exp(-(k-k[npts//2])**2) # shifted
Ptrues=[Ptrue1,Ptrue2,Ptrue3]
Pkinds=['Gaussian','flat','skewed','shifted']

fig,axs=plt.subplots(3,npts,layout='tight',figsize=(15,10))
grns=cm.Greens( np.linspace(0.2, 0.8, npts))
orgs=cm.Oranges(np.linspace(0.2, 0.8, npts))
Wmatrix=Wmat(k,0.3)
focus=npts//3
for p,Ptr in enumerate(Ptrues): # different Ptrue cases ... each goes in its own column
   axs[0,p].plot(k,Ptrues[p])
   for j in range(3):
      axs[j,p].set_xlabel('k')
      axs[j,p].set_ylabel('power')
   axs[0,p].set_title("Ptrue(k')")
   axs[0,p].scatter([k[focus]],[Ptrues[focus]],m='*')
   axs[1,p].set_title("W(k,k')")
   axs[2,p].set_title("Pobs(k) for a "+Pkinds[p]+" P(k)")
   Pobs=Wmatrix@Ptrues[p]
   axs[2,p].plot(k,Pobs,c='C1')
   axs[2,p].scatter([k[focus]],[Pobs[focus]],c='C1')
   for i,ki in enumerate(k):
#       Wveccur=Wmat(k,0.3)
#       axs[1,p].plot(k+ki,Wveccur,color=grns[i])
      if i==focus:
         linew=5
      else:
         linew=1
      axs[1,p].plot(k,Wmatrix[i,:],color=grns[i],lw=linew) # NOT np.roll(k,i)
#     #   Pobs=Wveccur*Ptrues[p]
#       Pobs=Wveccur@Ptrues[p]
#       axs[2,p].plot(k+ki,Pobs,color=orgs[i])
#       axs[2,p].scatter([2*ki],[Pobs[i]],marker='*',color=orgs[i])
plt.show()
fig.savefig('window_toy_model.png')

##########################################################################
# NOW, LOOK AT A SKEWED WINDOW FUNCTION

def Wmatskew(kvec,sigma): 
   '''
   kvec = VECTOR of k-points of interest for this power spectrum
   '''
   k,kp=np.meshgrid(kvec,kvec)
   return (2*np.pi*sigma**2)**3*np.exp(-sigma**2*(k-kp)**2/2.)*(k-kp)