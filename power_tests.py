from power import *
import time
import numpy as np
from matplotlib import pyplot as plt

############## TEST SPH FWD
Lsurvey = 103
Npix = 200 # 150 looks ok for spherical (if a little stripy for cylindrical), but turning up to 200 means the lowest-k bin is always empty (for spherical and along both axes for cylindrical ... I think it's just b/c the log-spaced bins are so close together)
Nk = 14

plt.figure()
maxvals=0.0
for i in range(5):
    T = np.random.normal(loc=0.0, scale=1.0, size=(Npix,Npix,Npix))
    kfloors,vals=generate_P(T,"lin",Lsurvey,Nk)
    plt.scatter(kfloors,vals)
    maxvalshere=np.max(vals)
    if (maxvalshere>maxvals):
        maxvals=maxvalshere
plt.xlabel("k (1/Mpc)")
plt.ylabel("Power (K$^2$ Mpc$^3$)")
plt.title("Test white noise P(k) calc for Lsurvey,Npix,Nk={:4},{:4},{:4}".format(Lsurvey,Npix,Nk))
plt.ylim(0,1.2*maxvals)
plt.savefig("wn_sph.png",dpi=500)
plt.show() # WORKS AS OF 14:28 20.05.25

# assert(1==0), "fix sph first"
# ############## TEST CYL FWD
Nkpar=9 # 327
Nkperp=12 # 1010
# Nkpar=300
# Nkperp=100

nsubrow=3
nsubcol=3
vmin=np.infty
vmax=-np.infty
fig,axs=plt.subplots(nsubrow,nsubcol,figsize=(8,10))
for i in range(nsubrow):
    for j in range(nsubcol):
        T = np.random.normal(loc=0.0, scale=1.0, size=(Npix,Npix,Npix))
        k,vals=generate_P(T,"lin",Lsurvey,Nkpar,Nk1=Nkperp) 
        kpar,kperp=k
        kpargrid,kperpgrid=np.meshgrid(kpar,kperp,indexing="ij")
        im=axs[i,j].pcolor(kpargrid,kperpgrid,vals)
        axs[i,j].set_ylabel("$k_{||}$")
        axs[i,j].set_xlabel("$k_\perp$")
        axs[i,j].set_title("Realization {:2}".format(i*nsubrow+j))
        axs[i,j].set_aspect("equal")
        minval=np.min(vals)
        maxval=np.max(vals)
        if (minval<vmin):
            vmin=minval
        if (maxval>vmax):
            vmax=maxval
fig.colorbar(im,extend="both")
plt.suptitle("Test white noise P(kpar,kperp) calc for Lsurvey,Npix,Nkpar,Nkperp={:4},{:4},{:4},{:4}".format(Lsurvey,Npix,Nkpar,Nkperp))
plt.tight_layout()
plt.savefig("wn_cyl.png",dpi=500)
plt.show()

############# TESTS BWD
Lsurvey=100 # Mpc
plot=True
cases=['ps_wn_2px.txt','z5spec.txt','ps_wn_20px.txt']
ncases=len(cases)
if plot:
    fig,axs=plt.subplots(2*ncases,3, figsize=(15,10)) # (3 power specs * 2 voxel schemes per power spec) = 6 generated boxes to look at slices of
t0=time.time()
for k,case in enumerate(cases):
    kfl,P=np.genfromtxt(case,dtype='complex').T
    Npix=len(P)

    # n_field_voxel_cases=[4,3]
    # n_field_voxel_cases=[21,22]
    # n_field_voxel_cases=[44,45] # 15 s
    # n_field_voxel_cases=[65,66] # 24 s
    # n_field_voxel_cases=[88,89] # 36 s
    n_field_voxel_cases=[99,100] # 97 s
    for j,n_field_voxels in enumerate(n_field_voxel_cases):
        tests=[0,n_field_voxels//2,n_field_voxels-3]
        rgen,Tgen,rmags=generate_box(P,kfl,Lsurvey,n_field_voxels)
        print('done with inversion for k,j=',k,j)
        if plot:
            for i,test in enumerate(tests):

                if len(cases)>1:
                    im=axs[2*k+j,i].imshow(Tgen[:,:,test])
                    fig.colorbar(im)
                    axs[2*k+j,i].set_title('slice '+str(test)+'/'+str(n_field_voxels)+'; original box = '+str(case))
                else:
                    im=axs[2*k+j,i].imshow(Tgen[:,:,test])
                    fig.colorbar(im)
                    axs[2*k+j,i].set_title('slice '+str(test)+'/'+str(n_field_voxels)+'; original box = '+str(case))

if plot:
    plt.suptitle('brightness temp box slices generated from inverting a PS I calculated')
    plt.tight_layout()
    fig.savefig('generate_box_tests.png')
    t1=time.time()
    plt.show()

print('test suite took',t1-t0,'s')