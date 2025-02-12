import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
from scipy.special import erf

def Wmat(kvec,sigma): 
   k,kp=np.meshgrid(kvec,kvec)
   return 2*np.pi*sigma**2*np.exp(-sigma**2*(k-kp)**2)

def Wmatskew(kvec,sigma): 
   '''
   kvec = VECTOR of k-points of interest for this power spectrum
   '''
   k,kp=np.meshgrid(kvec,kvec)
   return 2*np.pi*sigma**2*np.exp(-sigma**2*(k-kp)**2)*(1+erf(5*k))

def Pobs_wrapper(npts,klim,w,wsig,psig,figname):
   k=np.linspace(-klim,klim,npts)
   amp=1529
   Ptrue1=amp*np.exp(-k**2/psig**2) # gaussian
   Ptrue2=amp*np.ones(npts) # flat
   Ptrue3=amp*k*np.exp(-k**2/psig**2)*(1+erf(5*k)) # skewed
   Ptrue4=amp*np.exp(-(k-k[npts//4])**2/psig**2) # shifted
   Ptrues=np.array([Ptrue1,Ptrue2,Ptrue3,Ptrue4])
   Pkinds=['Gaussian','flat','skewed','shifted']

   fig,axs=plt.subplots(3,Ptrues.shape[0],layout='tight',figsize=(15,10))
   grns=cm.Greens( np.linspace(0.2, 0.8, npts))
   Wmatrix=w(k,wsig)
   focus=8*npts//15
   for p,Ptr in enumerate(Ptrues): # Ptrue cases go in separate columns
      axs[0,p].plot(k,Ptrues[p])
      for j in range(3): # ingredients in the matrix product Pobs=W@Ptrue go in separate rows
         axs[j,p].set_xlabel('k')
         axs[j,p].set_ylabel('power')
      axs[0,p].set_title("Ptrue(k') for a "+Pkinds[p]+" P(k)")
      # print('k.shape=',k.shape,', Ptrues.shape=',Ptrues.shape,', focus=',focus)
      axs[0,p].scatter([k[focus]],[Ptrues[p,focus]],marker='*')
      axs[1,p].set_title("W(k,k') for a **3D** Gaussian beam")
      axs[2,p].set_title("Pobs(k) for a "+Pkinds[p]+" P(k)")
      Pobs=Wmatrix@Ptrues[p]
      axs[2,p].plot(k,Pobs,c='C1')
      for i,ki in enumerate(k):
         if i==focus:
            linew=5
         else:
            linew=1
         axs[1,p].plot(k,Wmatrix[i,:],color=grns[i],lw=linew) # NOT np.roll(k,i)
   plt.show()
   fig.savefig(figname)
   return None

Pobs_wrapper(100, 50, Wmat,     0.3, 5, 'window_regu_toy_model.png') # npts, klim, w, wsig, psig, figname
# Pobs_wrapper(100, 50, Wmatskew, 0.3, 5, 'window_skew_toy_model.png')