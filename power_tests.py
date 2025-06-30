from power import *
from bias_helper_fcns import custom_response
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
    fig,axs=plt.subplots(1,4,figsize=(20,5))
    maxvals=0.
    maxvals_mod0=0.
    maxvals_mod1=0.
    maxvals_mod2=0.
    Nrealiz=35
    colours=plt.cm.Blues(np.linspace(0.2,1,Nrealiz))

    vec=1/np.fft.fftshift(np.fft.fftfreq(Npix,d=Lsurvey/Npix)) # based on k_vec_for_box=twopi*np.fft.fftshift(np.fft.fftfreq(Nvox,d=Delta)) and r=2pi/k
    xgrid,ygrid,zgrid=np.meshgrid(vec,vec,vec,indexing="ij")
    # print("config space voxel scale = vec[1]-vec[0]=",vec[1]-vec[0])
    sigma02=1e3
    sigma12=10
    sigma22=0.5
    modulation0=np.exp(-(xgrid**2+ygrid**2+zgrid**2)/(2*sigma02)) # really wide in config space
    modulation1=np.exp(-(xgrid**2+ygrid**2+zgrid**2)/(2*sigma12)) # medium width in config space
    modulation2=np.exp(-(xgrid**2+ygrid**2+zgrid**2)/(2*sigma22)) # really narrow in config space
    allvals= np.zeros((Nk,Nrealiz))
    allvals0=np.zeros((Nk,Nrealiz))
    allvals1=np.zeros((Nk,Nrealiz))
    allvals2=np.zeros((Nk,Nrealiz))

    for i in range(Nrealiz):
        T = np.random.normal(loc=0.0, scale=1.0, size=(Npix,Npix,Npix))
        Tmod0=T*modulation0
        # print("broad in config")
        kfloors_mod,vals_mod0=generate_P(Tmod0,mode,Lsurvey,Nk,custom_estimator=custom_response,custom_estimator_args=(np.sqrt(sigma02),np.sqrt(sigma02),np.sqrt(sigma02),2*np.sqrt(np.log(2)),)) # ,sigLoS,beamfwhm_x,beamfwhm_y,r0)
        allvals0[:,i]=vals_mod0
        Tmod1=T*modulation1
        # print("medium in config")
        kfloors_mod,vals_mod1=generate_P(Tmod1,mode,Lsurvey,Nk,custom_estimator=custom_response,custom_estimator_args=(np.sqrt(sigma12),np.sqrt(sigma12),np.sqrt(sigma12),2*np.sqrt(np.log(2)),))
        allvals1[:,i]=vals_mod1
        Tmod2=T*modulation2
        # print("narrow in config")
        kfloors_mod,vals_mod2=generate_P(Tmod2,mode,Lsurvey,Nk,custom_estimator=custom_response,custom_estimator_args=(np.sqrt(sigma22),np.sqrt(sigma22),np.sqrt(sigma22),2*np.sqrt(np.log(2)),))
        allvals2[:,i]=vals_mod2
        kfloors,vals=generate_P(T,mode,Lsurvey,Nk)
        allvals[:,i]=vals
        axs[0].scatter(kfloors,vals,color=colours[i])
        maxvalshere=np.max(vals)
        if (maxvalshere>maxvals):
            maxvals=maxvalshere
        axs[1].scatter(kfloors_mod,vals_mod0,color=colours[i])
        maxvalshere_mod0=np.max(vals_mod0)
        if (maxvalshere_mod0>maxvals_mod0):
            maxvals_mod0=maxvalshere_mod0
        maxvalshere_mod1=np.max(vals_mod1)
        if (maxvalshere_mod1>maxvals_mod1):
            maxvals_mod1=maxvalshere_mod1
        maxvalshere_mod2=np.max(vals_mod2)
        if (maxvalshere_mod2>maxvals_mod2):
            maxvals_mod2=maxvalshere_mod2
        axs[2].scatter(kfloors_mod,vals_mod1,color=colours[i])
        axs[3].scatter(kfloors_mod,vals_mod2,color=colours[i])
    for i in range(4):
        axs[i].set_xlabel("k (1/Mpc)")
        axs[i].set_ylabel("Power (K$^2$ Mpc$^3$)")
    axs[0].plot(kfloors,np.mean(allvals,axis=-1))
    axs[1].plot(kfloors,np.mean(allvals0,axis=-1))
    axs[2].plot(kfloors,np.mean(allvals1,axis=-1))
    axs[3].plot(kfloors,np.mean(allvals2,axis=-1))
    axs[0].set_title("P(T) / power spec of unmodulated box")
    axs[1].set_title("P(T*R) / power spec of response-modulated box \n(broad in config space)")
    axs[2].set_title("P(T*R) / power spec of response-modulated box \n(medium in config space)")
    axs[3].set_title("P(T*R) / power spec of response-modulated box \n(narrow in config space)")
    plt.suptitle("Test white noise P(k) calc for Lsurvey,Npix,Nk,sigma0**2,sigma1**2,sigma2**2={:4},{:4},{:4},{:4},{:4},{:4}".format(Lsurvey,Npix,Nk,sigma02,sigma12,sigma22))
    axs[0].set_ylim(0,1.2*maxvals)
    axs[1].set_ylim(0,1.2*maxvals_mod0)
    axs[2].set_ylim(0,1.2*maxvals_mod1)
    axs[3].set_ylim(0,1.2*maxvals_mod2)
    plt.tight_layout()
    plt.savefig("wn_sph_"+mode+".png",dpi=500)
    plt.show()

test_sph_interp=False
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


    ##
    fig,axs=plt.subplots(4,4,figsize=(20,20))
    maxvals=0.
    maxvals_mod0=0.
    maxvals_mod1=0.
    maxvals_mod2=0.
    Nrealiz=35

    vec=1/np.fft.fftshift(np.fft.fftfreq(Npix,d=Lsurvey/Npix)) # based on k_vec_for_box=twopi*np.fft.fftshift(np.fft.fftfreq(Nvox,d=Delta)) and r=2pi/k
    xgrid,ygrid,zgrid=np.meshgrid(vec,vec,vec,indexing="ij")
    # print("config space voxel scale = vec[1]-vec[0]=",vec[1]-vec[0])
    sigma02=1e3
    sigma12=10
    sigma22=0.5
    modulation0=np.exp(-(xgrid**2+ygrid**2+zgrid**2)/(2*sigma02)) # really wide in config space
    modulation1=np.exp(-(xgrid**2+ygrid**2+zgrid**2)/(2*sigma12)) # medium width in config space
    modulation2=np.exp(-(xgrid**2+ygrid**2+zgrid**2)/(2*sigma22)) # really narrow in config space
    allvals= np.zeros((Nkpar,Nkperp,Nrealiz))
    allvals0=np.zeros((Nkpar,Nkperp,Nrealiz))
    allvals1=np.zeros((Nkpar,Nkperp,Nrealiz))
    allvals2=np.zeros((Nkpar,Nkperp,Nrealiz))

    for i in range(Nrealiz):
        T = np.random.normal(loc=0.0, scale=1.0, size=(Npix,Npix,Npix))
        Tmod0=T*modulation0
        # print("broad in config")
        kfloors_mod,vals_mod0=generate_P(Tmod0,mode,Lsurvey,Nkpar,Nk1=Nkperp,custom_estimator=custom_response,custom_estimator_args=(np.sqrt(sigma02),np.sqrt(sigma02),np.sqrt(sigma02),2*np.sqrt(np.log(2)),)) # ,sigLoS,beamfwhm_x,beamfwhm_y,r0)
        # print("vals_mod0.shape=",vals_mod0.shape)
        allvals0[:,:,i]=vals_mod0
        Tmod1=T*modulation1
        # print("medium in config")
        kfloors_mod,vals_mod1=generate_P(Tmod1,mode,Lsurvey,Nkpar,Nk1=Nkperp,custom_estimator=custom_response,custom_estimator_args=(np.sqrt(sigma12),np.sqrt(sigma12),np.sqrt(sigma12),2*np.sqrt(np.log(2)),))
        allvals1[:,:,i]=vals_mod1
        Tmod2=T*modulation2
        # print("narrow in config")
        kfloors_mod,vals_mod2=generate_P(Tmod2,mode,Lsurvey,Nkpar,Nk1=Nkperp,custom_estimator=custom_response,custom_estimator_args=(np.sqrt(sigma22),np.sqrt(sigma22),np.sqrt(sigma22),2*np.sqrt(np.log(2)),))
        allvals2[:,:,i]=vals_mod2
        kfloors,vals=generate_P(T,mode,Lsurvey,Nkpar,Nk1=Nkperp,custom_estimator=custom_response,custom_estimator_args=(np.sqrt(sigma22),np.sqrt(sigma22),np.sqrt(sigma22),2*np.sqrt(np.log(2)),))
        allvals[:,:,i]=vals
    valmin=np.min(allvals)
    valmax=np.max(allvals)
    kparfloors,kperpfloors=kfloors
    kparfloorsgrid,kperpfloorsgrid=np.meshgrid(kparfloors,kperpfloors,indexing="ij")

    column_names=["unmod","mod broad in config sp","mod medium in config sp","mod narrow in config sp"]
    row_names=["realiz 0","realiz 1","realiz 2","average of"+str(Nrealiz)+"realix ("+str(Nrealiz-3)+"realiz not shown)"]
    for i in range(4):
        for j in range(4):
            axs[i,j].set_xlabel("k (1/Mpc)")
            axs[i,j].set_ylabel("P (K$^2$ Mpc$^3$)")
            axs[i,j].set_title(column_names[j]+" - "+row_names[i])
        if (i<3):
            # print("kparfloors.shape,kperpfloors.shape,allvals[:,:,i].shape=",kparfloors.shape,kperpfloors.shape,allvals[:,:,i].shape)
            im=axs[i,0].pcolor(kparfloorsgrid,kperpfloorsgrid,allvals[:,:,i],  vmin=valmin,vmax=valmax)
            fig.colorbar(im,ax=axs[i,0])
            im=axs[i,1].pcolor(kparfloorsgrid,kperpfloorsgrid,allvals0[:,:,i], vmin=valmin,vmax=valmax)
            fig.colorbar(im,ax=axs[i,1])
            im=axs[i,2].pcolor(kparfloorsgrid,kperpfloorsgrid,allvals1[:,:,i], vmin=valmin,vmax=valmax)
            fig.colorbar(im,ax=axs[i,2])
            im=axs[i,3].pcolor(kparfloorsgrid,kperpfloorsgrid,allvals2[:,:,i], vmin=valmin,vmax=valmax)
            fig.colorbar(im,ax=axs[i,3])

        

    im=axs[3,0].pcolor(kparfloorsgrid,kperpfloorsgrid,np.mean(allvals,axis=-1),  vmin=valmin,vmax=valmax)
    fig.colorbar(im,ax=axs[3,0])
    im=axs[3,1].pcolor(kparfloorsgrid,kperpfloorsgrid,np.mean(allvals0,axis=-1), vmin=valmin,vmax=valmax)
    fig.colorbar(im,ax=axs[3,0])
    im=axs[3,2].pcolor(kparfloorsgrid,kperpfloorsgrid,np.mean(allvals1,axis=-1), vmin=valmin,vmax=valmax)
    fig.colorbar(im,ax=axs[3,0])
    im=axs[3,3].pcolor(kparfloorsgrid,kperpfloorsgrid,np.mean(allvals2,axis=-1), vmin=valmin,vmax=valmax)
    fig.colorbar(im,ax=axs[3,0])
    # fig.colorbar(im)

    plt.suptitle("Test white noise P(kpar,kperp) calc for Lsurvey,Npix,Nkpar,Nkperp,sigma0**2,sigma1**2,sigma2**2={:4},{:4},{:4},{:4},{:4},{:4},{:4}".format(Lsurvey,Npix,Nkpar,Nkperp,sigma02,sigma12,sigma22))
    
    plt.tight_layout()
    plt.savefig("wn_cyl_"+mode+".png",dpi=500)
    plt.show()

test_cyl_interp=False
if test_cyl_interp:
    print("CYL INTERP")
    T = np.random.normal(loc=0.0, scale=1.0, size=(Npix,Npix,Npix))

    ### start of modulation test
    vec=1/np.fft.fftshift(np.fft.fftfreq(Npix,d=Lsurvey/Npix)) # based on k_vec_for_box=twopi*np.fft.fftshift(np.fft.fftfreq(Nvox,d=Delta)) and r=2pi/k
    print("vec=",vec)
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

test_bwd=False
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