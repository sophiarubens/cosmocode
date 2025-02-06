import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
# from power import *

def Wvec(k,kpi,sigma): 
   '''
   k   = VECTOR of k-points of interest for this power spectrum
   kpi = ith ELEMENT of kprime (where you're interested in sampling the window function)
   '''
   return (2*np.pi*sigma**2)**3*np.exp(-sigma**2*(k-kpi)**2/2.)

npts=10
# Pobs(k) = W(k,kpr)Ptrue(kpr)
# ultimately interested in Pobs(k)
# start by doing something similar to what Hannah did in the ApJ paper:
# plot Ptrue(kpr) with a shared-axis plot below showing W(k,kpr)
k=np.linspace(0.05,0.7,npts)
amp=1529 # smarter choice??

Ptrue1=amp*np.exp(-k**2) # gaussian
Ptrue2=amp*np.ones(npts) # flat
Ptrue3=amp*k*np.exp(-k**2) # skewed
Ptrues=[Ptrue1,Ptrue2,Ptrue3]
nps=len(Ptrues)

fig,axs=plt.subplots(3,nps,layout='constrained')
grns=cm.Greens(np.linspace(0.2, 0.8, npts))
orgs=cm.Oranges(np.linspace(0.2, 0.8, npts))
for p,Ptr in enumerate(Ptrues): # different Ptrue cases ... each goes in its own column
   axs[0,p].plot(k,Ptrues[p])
   for j in range(3):
      axs[j,p].set_xlabel('k')
      axs[j,p].set_ylabel('power')
   axs[0,p].set_title("Ptrue(k')")
   axs[1,p].set_title("W(k,k')")
   axs[2,p].set_title("Pobs(k)")
   for i,ki in enumerate(k):
      Wveccur=Wvec(k,ki,0.3)
      axs[1,p].plot(k+ki,Wveccur,c=grns[i])
      Pobs=Wveccur*Ptrues[p]
      axs[2,p].plot(k+ki,Pobs,c=orgs[i])
      axs[2,p].scatter([2*ki],[Pobs[i]],marker='*',c=orgs[i])
plt.show()
fig.savefig('window_toy_model.png')

##########################################################################
# NOW, LOOK AT A SKEWED WINDOW FUNCTION

def Wvec_skew(k,kpi,sigma): 
   '''
   k   = VECTOR of k-points of interest for this power spectrum
   kpi = ith ELEMENT of kprime (where you're interested in sampling the window function)
   '''
   return (2*np.pi*sigma**2)**3*np.exp(-sigma**2*(k-kpi)**2/2.)*(k-kpi)

npts=10
# Pobs(k) = W(k,kpr)Ptrue(kpr)
# ultimately interested in Pobs(k)
# start by doing something similar to what Hannah did in the ApJ paper:
# plot Ptrue(kpr) with a shared-axis plot below showing W(k,kpr)

fig,axs=plt.subplots(3,nps,layout='constrained')
grns=cm.Greens(np.linspace(0.2, 0.8, npts))
orgs=cm.Oranges(np.linspace(0.2, 0.8, npts))
for p,Ptr in enumerate(Ptrues): # different Ptrue cases ... each goes in its own column
   axs[0,p].plot(k,Ptrues[p])
   for j in range(3):
      axs[j,p].set_xlabel('k')
      axs[j,p].set_ylabel('power')
   axs[0,p].set_title("Ptrue(k')")
   axs[1,p].set_title("W(k,k')")
   axs[2,p].set_title("Pobs(k)")
   for i,ki in enumerate(k):
      Wveccur=Wvec_skew(k,ki,0.3)
      axs[1,p].plot(k+ki,Wveccur,c=grns[i])
      Pobs=Wveccur*Ptrues[p]
      axs[2,p].plot(k+ki,Pobs,c=orgs[i])
      axs[2,p].scatter([2*ki],[Pobs[i]],marker='*',c=orgs[i])
plt.show()
fig.savefig('window_skewed_toy_model.png')