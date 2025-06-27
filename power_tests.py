from power import *
import time
import numpy as np
from matplotlib import pyplot as plt
Lsurvey=126
Npix=52
mode="lin"
# mode="log"
Nkpar=8 # 327
Nkperp=12 # 1010
Nk = 14

test_sph_fwd=True
if test_sph_fwd:
    plt.figure()
    maxvals=0.0
    for i in range(5):
        T = np.random.normal(loc=0.0, scale=1.0, size=(Npix,Npix,Npix))
        kfloors,vals=generate_P(T,mode,Lsurvey,Nk)
        plt.scatter(kfloors,vals)
        maxvalshere=np.max(vals)
        if (maxvalshere>maxvals):
            maxvals=maxvalshere
    plt.xlabel("k (1/Mpc)")
    plt.ylabel("Power (K$^2$ Mpc$^3$)")
    plt.title("Test white noise P(k) calc for Lsurvey,Npix,Nk={:4},{:4},{:4}".format(Lsurvey,Npix,Nk))
    plt.ylim(0,1.2*maxvals)
    plt.savefig("wn_sph_"+mode+".png",dpi=500)
    plt.show()

test_sph_interp=True
if test_sph_interp:
    T = np.random.normal(loc=0.0, scale=1.0, size=(Npix,Npix,Npix))
    kfloors,vals=generate_P(T,mode,Lsurvey,Nk)
    k_want_lo=0.01
    k_want_hi=4.
    k_want=np.linspace(k_want_lo,k_want_hi,3*Nk)
    k_have=np.reshape(kfloors,(Nk,))
    P_have=np.reshape(vals,(Nk,))
    k_want_returned,P_want=interpolate_P(P_have,k_have,k_want,avoid_extrapolation=False)
    plt.figure()
    plt.scatter(kfloors,        vals,  label="generated P(k)")
    plt.scatter(k_want_returned,P_want,label="interpolated P(k)")
    plt.xlabel("k")
    plt.ylabel("P(k)")
    plt.title("spherical P(k) comparison")
    plt.axvline(kfloors[0], c="C0")
    plt.axvline(kfloors[-1],c="C0",label="extent of generated P")
    plt.axvline(k_want_lo,  c="C1")
    plt.axvline(k_want_hi,  c="C1",label="extent of interpolated P")
    plt.legend()
    plt.savefig("sph_interp_"+mode+".png",dpi=500)
    plt.show()

test_cyl_fwd=True
if test_cyl_fwd:
    nsubrow=3
    nsubcol=3
    vmin=np.infty
    vmax=-np.infty

    ###
    vec=1/np.fft.fftshift(np.fft.fftfreq(Npix,d=Lsurvey/Npix)) # based on k_vec_for_box=twopi*np.fft.fftshift(np.fft.fftfreq(Nvox,d=Delta)) and r=2pi/k
    xgrid,ygrid,zgrid=np.meshgrid(vec,vec,vec,indexing="ij")
    modulation=np.exp(-(xgrid**2+ygrid**2+zgrid**2)/2)
    plt.figure()
    slice=3
    plt.imshow(modulation[:,:,slice])
    plt.title("modulation[:,:,"+str(slice)+"]")
    plt.show()
    ###

    fig,axs=plt.subplots(nsubrow,nsubcol,figsize=(8,10))

    for i in range(nsubrow):
        for j in range(nsubcol):
            T = np.random.normal(loc=0.0, scale=1.0, size=(Npix,Npix,Npix))

            ### start of modulation test
            Tmod=T*modulation
            k,vals=generate_P(Tmod,mode,Lsurvey,Nkpar,Nk1=Nkperp)
            ### end of modulation test

            # k,vals=generate_P(T,mode,Lsurvey,Nkpar,Nk1=Nkperp) 
            kpar,kperp=k
            kpargrid,kperpgrid=np.meshgrid(kpar,kperp,indexing="ij")
            im=axs[i,j].pcolor(kpargrid,kperpgrid,vals)
            axs[i,j].set_ylabel("$k_\perp$")
            axs[i,j].set_xlabel("$k_{||}$")
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
    plt.savefig("wn_cyl_"+mode+".png",dpi=500)
    plt.show()

test_cyl_interp=True
if test_cyl_interp:
    print("CYL INTERP")
    T = np.random.normal(loc=0.0, scale=1.0, size=(Npix,Npix,Npix))

    ### start of modulation test
    vec=1/np.fft.fftshift(np.fft.fftfreq(Npix,d=Lsurvey/Npix)) # based on k_vec_for_box=twopi*np.fft.fftshift(np.fft.fftfreq(Nvox,d=Delta)) and r=2pi/k
    xgrid,ygrid,zgrid=np.meshgrid(vec,vec,vec,indexing="ij")
    modulation=np.exp(-(xgrid**2+ygrid**2+zgrid**2))
    Tmod=T*modulation
    k,vals=generate_P(Tmod,mode,Lsurvey,Nkpar,Nk1=Nkperp)
    ### end of modulation test

    # k,vals=generate_P(T,mode,Lsurvey,Nkpar,Nk1=Nkperp)
    kpar_have,kperp_have=k
    kpar_have_grid,kperp_have_grid=np.meshgrid(kpar_have,kperp_have,indexing="ij")
    kpar_want=np.linspace( 0.1866,  0.9702, 2*Nkpar)
    kperp_want=np.linspace(0.08485, 1.290,  2*Nkperp)
    k_want=(kpar_want,kperp_want)
    k_want_returned,P_want=interpolate_P(vals,k,k_want,avoid_extrapolation=False)
    kpar_want_returned,kperp_want_returned=k_want_returned
    # print("np.all(kpar_want_returned==kpar_want),np.all(kperp_want_returned==kperp_want)=",np.all(kpar_want_returned==kpar_want),np.all(kperp_want_returned==kperp_want)) # PASSES
    kpar_want_grid,kperp_want_grid=np.meshgrid(kpar_want_returned,kperp_want_returned,indexing="ij")
    fig,axs=plt.subplots(1,2,figsize=(10,5))
    im=axs[0].pcolor(kpar_have_grid,kperp_have_grid,vals)
    axs[0].set_title("P from generate_P")
    im=axs[1].pcolor(kpar_want_grid,kperp_want_grid,P_want)
    axs[1].axvline(kpar_have[0],   c="C0")
    axs[1].axvline(kpar_have[-1],  c="C0")
    axs[1].axhline(kperp_have[0],  c="C0")
    axs[1].axhline(kperp_have[-1], c="C0", label="extent of original P")
    axs[1].axvline(kpar_want[0],   c="C1")
    axs[1].axvline(kpar_want[-1],  c="C1")
    axs[1].axhline(kperp_want[0],  c="C1")
    axs[1].axhline(kperp_want[-1], c="C1",label="extent of interpolated P")
    axs[1].set_title("interpolated P")
    minval=np.min((np.min(vals),np.min(P_want)))
    maxval=np.max((np.max(vals),np.max(P_want)))
    plt.colorbar(im,extend="both")
    for i in range(2):
        axs[i].set_xlabel("kpar")
        axs[i].set_ylabel("kperp")
    plt.legend(loc="upper right")
    plt.suptitle("power spectrum interpolation tests")
    plt.savefig("cyl_interp_"+mode+".png")
    plt.show()

test_bwd=True
if test_bwd:
    Lsurvey=103 # Mpc
    plot=True
    cases=['ps_wn_2px.txt','z5spec.txt','ps_wn_20px.txt']
    ncases=len(cases)
    if plot:
        fig,axs=plt.subplots(2*ncases,3, figsize=(15,10)) # (3 power specs * 2 voxel schemes per power spec) = 6 generated boxes to look at slices of
    t0=time.time()
    for k,case in enumerate(cases):
        kfl,P=np.genfromtxt(case,dtype='complex').T
        Npix=len(P)

        n_field_voxel_cases=[99,100] # 4.8 s for the whole loop
        # n_field_voxel_cases=[199,200] # 22.2 s for the whole loop
        # n_field_voxel_cases=[399,400] # 172.5 s for the whole loop
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