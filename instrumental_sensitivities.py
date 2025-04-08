import numpy as np
from matplotlib import pyplot as plt 
from cosmo_distances import *

nu21=1420.405751768 # MHz
npts=100
z_values=np.linspace(0,15,npts)
nu_values=z2freq(nu21,z_values)
comoving_distances=np.zeros(npts)
for i,z in enumerate(z_values):
    comoving_distances[i]=comoving_distance(z)
zHERA=np.array([6,12])
zCHORD=np.array([0,3.5])
zCHIME=np.array([0.8,2.5])
zCHIME21=np.array([0.78,1.43])
zCHORD21hyp=np.array([0.45,0.75])
zObservatories=np.asarray([zHERA,zCHORD,zCHIME,zCHIME21,zCHORD21hyp])
nuHERA=z2freq(nu21,zHERA)
nuCHORD=z2freq(nu21,zCHORD)
nuCHIME=z2freq(nu21,zCHIME)
nuCHIME21=z2freq(nu21,zCHIME21)
nuCHORD21hyp=z2freq(nu21,zCHORD21hyp)
nuObservatories=np.asarray([nuHERA,nuCHORD,nuCHIME,nuCHIME21,nuCHORD21hyp])
observatories=['HERA','CHORD','CHIME','CHIME 21 cm','hypothetical CHORD 21 cm']

alpha_use=0.7
fig,axs=plt.subplots(1,2,figsize=(15,5))
axs[0].plot(z_values,comoving_distances)
axs[0].set_xlabel('redshift')
axs[0].set_ylabel('comoving distance (Mpc)')
axs[0].set_title('redshift-space curve')
axs[1].plot(nu_values,comoving_distances)
axs[1].set_xlabel('observed frequency')
axs[1].set_ylabel('comoving distance (Mpc)')
axs[1].set_title('frequency-space curve')
for i,zObservatory in enumerate(zObservatories):
    nuObservatory=nuObservatories[i]
    print('observatories[i],nuObservatory,zObservatory=',observatories[i],nuObservatory,zObservatory)
    axs[0].fill_betweenx(comoving_distances,zObservatory[0],zObservatory[1],label=observatories[i],alpha=alpha_use)
    axs[1].fill_betweenx(comoving_distances,nuObservatory[0],nuObservatory[1],label=observatories[i],alpha=alpha_use)
axs[0].legend()
axs[1].legend()
plt.suptitle("Frequency and redshift windows for various observatories which may conduct LIM surveys")
plt.savefig("observing_regimes.png")
plt.tight_layout()
plt.show()

fig,axs=plt.subplots(1,2,figsize=(15,5))
axs[0].plot(z_values,comoving_distances)
axs[0].set_xlabel('redshift')
axs[0].set_ylabel('comoving distance (Mpc)')
axs[0].set_title('redshift-space curve')
axs[1].plot(nu_values,comoving_distances)
axs[1].set_xlabel('observed frequency')
axs[1].set_ylabel('comoving distance (Mpc)')
axs[1].set_title('frequency-space curve')
for i,zObservatory in enumerate(zObservatories):
    nuObservatory=nuObservatories[i]
    print('observatories[i],nuObservatory,zObservatory=',observatories[i],nuObservatory,zObservatory)
    # lointerceptx=nuObservatory[0]
    lointercepty=comoving_distance(zObservatory[0])
    # hiinterceptx=nuObservatory[1]
    hiintercepty=comoving_distance(zObservatory[1])
    axs[0].fill_between(z_values, lointercepty,hiintercepty,label=observatories[i],alpha=alpha_use) #fill_between(x, y1, y2=0, where=None, interpolate=False, step=None, *, data=None, **kwargs)
    axs[1].fill_between(nu_values,lointercepty,hiintercepty,label=observatories[i],alpha=alpha_use)
axs[0].legend()
axs[1].legend()
plt.suptitle("LoS sensitivity for various observatories which may conduct LIM surveys")
plt.savefig("observing_shells.png")
plt.tight_layout()
plt.show()

# CHORD baselines range between 6 m and 245 m -> probe kperp from just below 0.1 to just below 0.4 