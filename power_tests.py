from power import *
from bias_helper_fcns import *
import time
import numpy as np
from matplotlib import pyplot as plt
import time

Lsurvey=126 # 63
Nvox=52 # 10 
Nk = 14 # 8
mode="lin"
# mode="log"
Nkpar=11
Nkperp=13
Nrealiz=100

def elbowy_power(k,a=0.96605,b=-0.8,c=1,a0=1,b0=5000):
    return c/(a0*k**(-a)+b0*k**(-b))

####################################################################################################################################################################################
test_sph_fwd=True
visualize_T_slices=False
plot_fiducial_scaled=False
# power_spec_type="wn"
power_spec_type="pl"
if test_sph_fwd:
    print("GENERATE SPH POWER SPEC FROM BOX")
    t0=time.time()
    maxvals=0.
    maxvals_mod0=0.
    maxvals_mod1=0.
    maxvals_mod2=0.
    colours=plt.cm.Blues(np.linspace(0.2,1,Nrealiz))

    vec=Lsurvey*np.fft.fftshift(np.fft.fftfreq(Nvox))
    xgrid,ygrid,zgrid=np.meshgrid(vec,vec,vec,indexing="ij")
    sigma02=1e3 # wide   in config space
    sigma12=10  # medium in config space
    sigma22=0.5 # narrow in config space
    beamfwhm_x=3.5 # kind of hacky but enough to stop it from entering the Delta-like case every single time (for the numerics as of 2025.07.07 09:47, the voxel scale comparison value is ~2.42)
    beamfwhm_y=np.copy(beamfwhm_x)
    sigLoS0=np.sqrt(2.*sigma02)
    sigLoS1=np.sqrt(2.*sigma12)
    sigLoS2=np.sqrt(2.*sigma22)
    r00=np.sqrt(np.log(2))*sigLoS0
    r10=np.sqrt(np.log(2))*sigLoS1
    r20=np.sqrt(np.log(2))*sigLoS2
    print("sigLoS0,sigLoS1,sigLoS2,r00,r10,r20=",sigLoS0,sigLoS1,sigLoS2,r00,r10,r20)
    bundled0=(sigLoS0,beamfwhm_x,beamfwhm_y,r00,)
    modulation0=custom_response(xgrid,ygrid,zgrid,sigLoS0,beamfwhm_x,beamfwhm_y,r00)
    bundled1=(sigLoS1,beamfwhm_x,beamfwhm_y,r10)
    modulation1=custom_response(xgrid,ygrid,zgrid,sigLoS1,beamfwhm_x,beamfwhm_y,r10)
    bundled2=(sigLoS2,beamfwhm_x,beamfwhm_y,r20,)
    modulation2=custom_response(xgrid,ygrid,zgrid,sigLoS2,beamfwhm_x,beamfwhm_y,r20)
    allvals= np.zeros((Nk,Nrealiz))
    allvals0=np.zeros((Nk,Nrealiz))
    allvals1=np.zeros((Nk,Nrealiz))
    allvals2=np.zeros((Nk,Nrealiz))

    scaleto=-1
    for i in range(Nrealiz):
        alert=Nrealiz//5
        if alert>0:
            if (i%alert==0):
                print("realization",i)
        if   power_spec_type=="wn":
            T = np.random.normal(loc=0.0, scale=1.0, size=(Nvox,Nvox,Nvox))
        elif power_spec_type=="pl":
            if (i==0):
                ktest=np.linspace(twopi/Lsurvey,twopi*Nvox/Lsurvey,Nk)
                idx=-0.96605
                Ptest=ktest**idx
            _,T, _=generate_box(Ptest,ktest,Lsurvey,Nvox) # generate_box(P,k,Lsurvey,Nvox,primary_beam=False,primary_beam_args=False)
            _,T0,_=generate_box(Ptest,ktest,Lsurvey,Nvox, primary_beam=custom_response,primary_beam_args=bundled0)
            _,T1,_=generate_box(Ptest,ktest,Lsurvey,Nvox, primary_beam=custom_response,primary_beam_args=bundled1)
            _,T2,_=generate_box(Ptest,ktest,Lsurvey,Nvox, primary_beam=custom_response,primary_beam_args=bundled2)
        if i==0:
            V=Lsurvey**3
            if visualize_T_slices:
                figfig,axsaxs=plt.subplots(3,4,figsize=(20,10))
                qtr=Nvox//4
                hlf=Nvox//2
                slices=[0,qtr,hlf,-3]
                for i,slice in enumerate(slices):
                    axsaxs[0,i].imshow(T[:,:,slice])
                    axsaxs[0,i].set_title("T[:,:,"+str(slice)+"]")
                    axsaxs[1,i].imshow(T[:,slice,:])
                    axsaxs[1,i].set_title("T[:,"+str(slice)+",:]")
                    axsaxs[2,i].imshow(T[slice,:,:])
                    axsaxs[2,i].set_title("T["+str(slice)+",:,:]")
                for i in range(3):
                    for j in range(4):
                        axsaxs[i,j].set_xlabel("voxel index")
                        axsaxs[i,j].set_ylabel("voxel index")
                plt.suptitle("inspect box slices for underpopulation issues using iteration 0")
                plt.tight_layout()
                plt.savefig("inspect_box_slices_for_potential_underpop.png")
                plt.show()
            # fig,axs=plt.subplots(1,4,figsize=(20,5))  # version with no inset for the reconstructed values
            # fig,axs=plt.subplots(2,4,figsize=(20,10)) # version with an inset for the reconstructed values
            fig,axs=plt.subplots(3,4,figsize=(20,15)) # version that adds ratios of the fiducial and reconstructed values
        Tmod0=T0*modulation0
        kfloors_mod,vals_mod0=generate_P(Tmod0,mode,Lsurvey,Nk, primary_beam=custom_response,primary_beam_args=bundled0) # generate_P(T, mode, Lsurvey, Nk0, Nk1=0, primary_beam=False,primary_beam_args=False) 
        allvals0[:,i]=vals_mod0
        Tmod1=T1*modulation1
        kfloors_mod,vals_mod1=generate_P(Tmod1,mode,Lsurvey,Nk, primary_beam=custom_response,primary_beam_args=bundled1)
        allvals1[:,i]=vals_mod1
        Tmod2=T2*modulation2
        kfloors_mod,vals_mod2=generate_P(Tmod2,mode,Lsurvey,Nk, primary_beam=custom_response,primary_beam_args=bundled2)
        allvals2[:,i]=vals_mod2
        kfloors,vals=generate_P(T,mode,Lsurvey,Nk)
        allvals[:,i]=vals
        axs[0,0].scatter(kfloors,vals,color=colours[i])
        axs[1,0].scatter(kfloors,vals,color=colours[i])
        maxvalshere=np.max(vals)
        if (maxvalshere>maxvals):
            maxvals=maxvalshere
        axs[0,1].scatter(kfloors_mod,vals_mod0,color=colours[i])
        axs[1,1].scatter(kfloors_mod,vals_mod0,color=colours[i])
        maxvalshere_mod0=np.max(vals_mod0)
        if (maxvalshere_mod0>maxvals_mod0):
            maxvals_mod0=maxvalshere_mod0
        maxvalshere_mod1=np.max(vals_mod1)
        if (maxvalshere_mod1>maxvals_mod1):
            maxvals_mod1=maxvalshere_mod1
        maxvalshere_mod2=np.max(vals_mod2)
        if (maxvalshere_mod2>maxvals_mod2):
            maxvals_mod2=maxvalshere_mod2
        axs[0,2].scatter(kfloors_mod,vals_mod1,color=colours[i])
        axs[0,3].scatter(kfloors_mod,vals_mod2,color=colours[i])
        axs[1,2].scatter(kfloors_mod,vals_mod1,color=colours[i])
        axs[1,3].scatter(kfloors_mod,vals_mod2,color=colours[i])
    for i in range(3):
        for j in range(4):
            ylabels=["Power (K$^2$ Mpc$^3$)","Power (K$^2$ Mpc$^3$)","Ratio of powers (dimensionless, unitless)"]
            axs[i,j].set_xlabel("k (1/Mpc)")
            axs[i,j].set_ylabel(ylabels[i])
    meanmean=np.mean(allvals,axis=-1)
    mean0=np.mean(allvals0,axis=-1)
    mean1=np.mean(allvals1,axis=-1)
    mean2=np.mean(allvals2,axis=-1)
    axs[0,0].plot(kfloors,meanmean, label="reconstructed")
    axs[0,1].plot(kfloors,mean0,label="reconstructed")
    axs[0,2].plot(kfloors,mean1,label="reconstructed")
    axs[0,3].plot(kfloors,mean2,label="reconstructed")
    axs[1,0].plot(kfloors,meanmean, label="reconstructed")
    axs[1,1].plot(kfloors,mean0,label="reconstructed")
    axs[1,2].plot(kfloors,mean1,label="reconstructed")
    axs[1,3].plot(kfloors,mean2,label="reconstructed")
    if (power_spec_type=="pl"):
        scale=1/kfloors[scaleto]**idx*meanmean[scaleto]
        scale0=1/kfloors[scaleto]**idx*mean0[scaleto]
        scale1=1/kfloors[scaleto]**idx*mean1[scaleto]
        scale2=1/kfloors[scaleto]**idx*mean2[scaleto]
        if plot_fiducial_scaled:
            axs[0,0].plot(kfloors,kfloors**idx*scale, label="fiducial scaled")
            axs[0,1].plot(kfloors,kfloors**idx*scale0,label="fiducial scaled")
            axs[0,2].plot(kfloors,kfloors**idx*scale1,label="fiducial scaled")
            axs[0,3].plot(kfloors,kfloors**idx*scale2,label="fiducial scaled")
            axs[1,0].plot(kfloors,kfloors**idx*scale, label="fiducial scaled")
            axs[1,1].plot(kfloors,kfloors**idx*scale0,label="fiducial scaled")
            axs[1,2].plot(kfloors,kfloors**idx*scale1,label="fiducial scaled")
            axs[1,3].plot(kfloors,kfloors**idx*scale2,label="fiducial scaled")
        print("scale,scale0,scale1,scale2=",scale,scale0,scale1,scale2)
        for i in range(2):
            for j in range(4):
                axs[i,j].plot(kfloors,kfloors**idx,label="fiducial")
    axs[0,0].set_title("P(T) / power spec of unmodulated box")
    axs[0,1].set_title("P(T*R) / power spec of response-modulated box \n(broad in config space)")
    axs[0,2].set_title("P(T*R) / power spec of response-modulated box \n(medium in config space)")
    axs[0,3].set_title("P(T*R) / power spec of response-modulated box \n(narrow in config space)")
    for j in range(4):
        axs[1,j].set_title("inset for the case above")
    
    axs[2,0].plot(kfloors,kfloors**idx/meanmean)
    axs[2,1].plot(kfloors,kfloors**idx/mean0)
    axs[2,2].plot(kfloors,kfloors**idx/mean1)
    axs[2,3].plot(kfloors,kfloors**idx/mean2)
    for j in range(4):
        axs[2,j].set_title("Ratio of powers: fiducial/reconstructed")
    
    plt.suptitle("Test {:4} P(k) calc for Lsurvey,Nvox,Nk,Nrealiz,sigma0**2,sigma1**2,sigma2**2={:4},{:4},{:4},{:4},{:4},{:4}".format(power_spec_type,Lsurvey,Nvox,Nk,Nrealiz,sigma02,sigma12,sigma22))
    axs[0,0].set_ylim(0,1.2*maxvals)
    axs[0,1].set_ylim(0,1.2*maxvals_mod0)
    axs[0,2].set_ylim(0,1.2*maxvals_mod1)
    axs[0,3].set_ylim(0,1.2*maxvals_mod2)
    axs[1,0].set_ylim(0,1.2*meanmean[0])
    axs[1,1].set_ylim(0,1.2*mean0[0])
    axs[1,2].set_ylim(0,1.2*mean1[0])
    axs[1,3].set_ylim(0,1.2*mean2[0])
    plt.legend()
    plt.tight_layout()
    plt.savefig(power_spec_type+"_sph_"+mode+"_"+str(Nrealiz)+"realiz.png",dpi=1000)
    t1=time.time()
    print("generating sph power spectra took",t1-t0,"s\n")
    plt.show()

####################################################################################################################################################################################
test_sph_interp=True
if test_sph_interp:
    print("INTERP SPH POWER SPEC")
    t0=time.time()
    T = np.random.normal(loc=0.0, scale=1.0, size=(Nvox,Nvox,Nvox))
    kfloors,vals=generate_P(T,mode,Lsurvey,Nk)
    k_want_lo=0.01
    k_want_hi=2.
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
    t1=time.time()
    plt.show()
    print("interpolating a sph power spectrum took",t1-t0,"s\n")

####################################################################################################################################################################################
test_cyl_fwd=True
# power_spec_type="wn"
# power_spec_type="bpl"
power_spec_type="pl"
if test_cyl_fwd:
    print("GENERATE CYL POWER SPEC FROM BOX")
    t0=time.time()
    fig,axs=plt.subplots(4,4,figsize=(20,20))
    maxvals=0.
    maxvals_mod0=0.
    maxvals_mod1=0.
    maxvals_mod2=0.
    Delta=Lsurvey/Nvox

    vec=Lsurvey*np.fft.fftshift(np.fft.fftfreq(Nvox))
    xgrid,ygrid,zgrid=np.meshgrid(vec,vec,vec,indexing="ij")
    sigma02=1e3
    sigma12=10
    sigma22=0.5

    ##
    beamfwhm_x=3.5 # kind of hacky but enough to stop it from entering the Delta-like case every single time (for the numerics as of 2025.07.07 09:47, the voxel scale comparison value is ~2.42)
    beamfwhm_y=np.copy(beamfwhm_x)
    sigLoS0=np.sqrt(2.*sigma02)
    sigLoS1=np.sqrt(2.*sigma12)
    sigLoS2=np.sqrt(2.*sigma22)
    r00=np.sqrt(np.log(2))*sigLoS0
    r10=np.sqrt(np.log(2))*sigLoS1
    r20=np.sqrt(np.log(2))*sigLoS2
    bundled0=(sigLoS0,beamfwhm_x,beamfwhm_y,r00,)
    modulation0=custom_response(xgrid,ygrid,zgrid,sigLoS0,beamfwhm_x,beamfwhm_y,r00)
    bundled1=(sigLoS1,beamfwhm_x,beamfwhm_y,r10)
    modulation1=custom_response(xgrid,ygrid,zgrid,sigLoS1,beamfwhm_x,beamfwhm_y,r10)
    bundled2=(sigLoS2,beamfwhm_x,beamfwhm_y,r20,)
    modulation2=custom_response(xgrid,ygrid,zgrid,sigLoS2,beamfwhm_x,beamfwhm_y,r20)
    allvals= np.zeros((Nkpar,Nkperp,Nrealiz))
    allvals0=np.zeros((Nkpar,Nkperp,Nrealiz))
    allvals1=np.zeros((Nkpar,Nkperp,Nrealiz))
    allvals2=np.zeros((Nkpar,Nkperp,Nrealiz))
    ##

    tprev=time.time()
    for i in range(Nrealiz):
        alert=Nrealiz//5
        if alert>0:
            if (i%alert==0):
                print("realization",i)
        if i==0:
            V=Lsurvey**3
            Veff0=get_Veff(custom_response,bundled0,Lsurvey,Nvox) # args need to be bundled as (sigLoS,beamfwhm_x,beamfwhm_y,r0,)
            Veff1=get_Veff(custom_response,bundled1,Lsurvey,Nvox)
            Veff2=get_Veff(custom_response,bundled2,Lsurvey,Nvox)
        if power_spec_type=="wn":
            T = np.random.normal(loc=0.0, scale=1.0, size=(Nvox,Nvox,Nvox))
        elif power_spec_type=="bpl":
            if (i==0):
                Npts_bpl=25
                k_bpl=np.linspace(twopi/Lsurvey,twopi/Delta,Npts_bpl)
                P_bpl=elbowy_power(k_bpl)
            _,T,_ = generate_box(P_bpl,k_bpl,Lsurvey,Nvox) # generate_box(P,k,Lsurvey,Nvox,verbose=False) returns rgrid,T,rmags
        elif power_spec_type=="pl":
            if (i==0):
                Npts_pl=25
                k_pl=np.linspace(twopi/Lsurvey,twopi/Delta,Npts_pl)
                P_pl=k_pl**(-0.96605)
            _,T,_=generate_box(P_pl,k_pl,Lsurvey,Nvox)
        else:
            assert(1==0), "unsupported power_spec_type"
        Tmod0=T*modulation0
        kfloors_mod,vals_mod0=generate_P(Tmod0,mode,Lsurvey,Nkpar,Nk1=Nkperp, primary_beam=custom_response,primary_beam_args=bundled0)
        allvals0[:,:,i]=vals_mod0
        Tmod1=T*modulation1
        kfloors_mod,vals_mod1=generate_P(Tmod1,mode,Lsurvey,Nkpar,Nk1=Nkperp, primary_beam=custom_response,primary_beam_args=bundled1)
        allvals1[:,:,i]=vals_mod1
        Tmod2=T*modulation2
        kfloors_mod,vals_mod2=generate_P(Tmod2,mode,Lsurvey,Nkpar,Nk1=Nkperp, primary_beam=custom_response,primary_beam_args=bundled2)
        allvals2[:,:,i]=vals_mod2
        kfloors,vals=generate_P(T,mode,Lsurvey,Nkpar,Nk1=Nkperp)
        allvals[:,:,i]=vals
    
    valmin=np.min(allvals)
    valmax=np.max(allvals)
    kparfloors,kperpfloors=kfloors
    kparfloorsgrid,kperpfloorsgrid=np.meshgrid(kparfloors,kperpfloors,indexing="ij")

    column_names=["unmod","mod broad in config sp","mod medium in config sp","mod narrow in config sp"]
    row_names=["realiz 0","realiz 1","realiz 2","\navg of "+str(Nrealiz)+" realiz \n("+str(Nrealiz-3)+"realiz not shown)"]

    maxunmod= np.max(allvals)
    max0=     np.max(allvals0)
    max1=     np.max(allvals1)
    max2=     np.max(allvals2)

    for i in range(4):
        for j in range(4):
            axs[i,j].set_xlabel("k (1/Mpc)")
            axs[i,j].set_ylabel("P (K$^2$ Mpc$^3$)")
            axs[i,j].set_title(column_names[j]+" - "+row_names[i])
        if (i<3):
            im=axs[i,0].pcolor(kparfloorsgrid,kperpfloorsgrid,allvals[:,:,i],vmin=0,vmax=maxunmod)
            fig.colorbar(im,ax=axs[i,0])
            im=axs[i,1].pcolor(kparfloorsgrid,kperpfloorsgrid,allvals0[:,:,i],vmin=0,vmax=max0)
            fig.colorbar(im,ax=axs[i,1])
            im=axs[i,2].pcolor(kparfloorsgrid,kperpfloorsgrid,allvals1[:,:,i],vmin=0,vmax=max1)
            fig.colorbar(im,ax=axs[i,2])
            im=axs[i,3].pcolor(kparfloorsgrid,kperpfloorsgrid,allvals2[:,:,i],vmin=0,vmax=max2)
            fig.colorbar(im,ax=axs[i,3])
 
    mean=  np.mean(allvals,axis=-1)
    mean0= np.mean(allvals0,axis=-1)
    mean1= np.mean(allvals1,axis=-1)
    mean2= np.mean(allvals2,axis=-1)
    im=axs[3,0].pcolor(kparfloorsgrid,kperpfloorsgrid,mean) #,vmin=0,vmax=maxunmod)
    fig.colorbar(im,ax=axs[3,0])
    im=axs[3,1].pcolor(kparfloorsgrid,kperpfloorsgrid,mean0) #,vmin=0,vmax=max0)
    fig.colorbar(im,ax=axs[3,1])
    im=axs[3,2].pcolor(kparfloorsgrid,kperpfloorsgrid,mean1) #,vmin=0,vmax=max1)
    fig.colorbar(im,ax=axs[3,2])
    im=axs[3,3].pcolor(kparfloorsgrid,kperpfloorsgrid,mean2) #,vmin=0,vmax=max2)
    fig.colorbar(im,ax=axs[3,3])

    plt.suptitle("Test "+power_spec_type+" P(kpar,kperp) calc for Lsurvey,Nvox,Nkpar,Nkperp,sigma0**2,sigma1**2,sigma2**2={:4},{:4},{:4},{:4},{:4},{:4},{:4}".format(Lsurvey,Nvox,Nkpar,Nkperp,sigma02,sigma12,sigma22))
    for i in range(4):
        for j in range(4):
            axs[i,j].set_aspect("equal")
    plt.tight_layout()
    plt.savefig(power_spec_type+"_cyl_mod_"+mode+"_"+str(Nrealiz)+"_realiz.png",dpi=500)
    t1=time.time()
    plt.show()
    print("constructing cyl power spectra took",t1-t0,"s\n")

####################################################################################################################################################################################
test_cyl_interp=True
if test_cyl_interp:
    print("INTERPOLATING CYL POWER SPEC")
    t0=time.time()
    T = np.random.normal(loc=0.0, scale=1.0, size=(Nvox,Nvox,Nvox))
    k,vals=generate_P(T,mode,Lsurvey,Nkpar,Nk1=Nkperp)

    kpar_have,kperp_have=k
    kpar_have_grid,kperp_have_grid=np.meshgrid(kpar_have,kperp_have,indexing="ij")
    kpar_want=np.linspace( 0.1866,  0.9702, 2*Nkpar)
    kperp_want=np.linspace(0.08485, 1.290,  2*Nkperp)
    k_want=(kpar_want,kperp_want)
    k_want_returned,P_want=interpolate_P(vals,k,k_want,avoid_extrapolation=False)
    kpar_want_returned,kperp_want_returned=k_want_returned
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
    t1=time.time()
    plt.show()
    print("interpolating a cyl power spectrum took",t1-t0,"s\n")

####################################################################################################################################################################################
test_bwd=True
if test_bwd:
    print("GENERATING BOXES FROM POWER SPECTRA")
    t0=time.time()
    cases=['ps_wn_2px.txt','z8spec.txt','ps_wn_20px.txt']
    ncases=len(cases)
    fig,axs=plt.subplots(2*ncases,3, figsize=(15,10)) # (3 power specs * 2 voxel schemes per power spec) = 6 generated boxes to look at slices of
    t0=time.time()
    for k,case in enumerate(cases):
        kfl,P=np.genfromtxt(case,dtype='complex').T
        Nvox=len(P)

        n_field_voxel_cases=[99,100] # 4.8 s for the whole loop; [199,200] # 22.2 s for the whole loop; [399,400] # 172.5 s for the whole loop
        for j,n_field_voxels in enumerate(n_field_voxel_cases):
            tests=[0,n_field_voxels//2,n_field_voxels-3]
            rgen,Tgen,rmags=generate_box(P,kfl,Lsurvey,n_field_voxels)
            print('done with inversion for k,j=',k,j)
            # if plot:
            for i,test in enumerate(tests):

                if len(cases)>1:
                    im=axs[2*k+j,i].imshow(Tgen[:,:,test])
                    fig.colorbar(im)
                    axs[2*k+j,i].set_title('slice '+str(test)+'/'+str(n_field_voxels)+'; original box = '+str(case))
                else:
                    im=axs[2*k+j,i].imshow(Tgen[:,:,test])
                    fig.colorbar(im)
                    axs[2*k+j,i].set_title('slice '+str(test)+'/'+str(n_field_voxels)+'; original box = '+str(case))

    # if plot:
    plt.suptitle('brightness temp box slices generated from inverting a PS I calculated')
    plt.tight_layout()
    fig.savefig('generate_box_tests.png')
    t1=time.time()
    plt.show()
    print('generating assorted boxes took',t1-t0,'s\n')