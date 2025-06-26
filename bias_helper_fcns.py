import numpy as np
import camb
from camb import model
from scipy.signal import convolve2d,convolve
from matplotlib import pyplot as plt
from power import *
import time

Omegam_Planck18=0.3158
Omegabh2_Planck18=0.022383
Omegach2_Planck18=0.12011
OmegaLambda_Planck18=0.6842
lntentenAS_Planck18=3.0448
tentenAS_Planck18=np.exp(lntentenAS_Planck18)
AS_Planck18=tentenAS_Planck18/10**10
ns_Planck18=0.96605
H0_Planck18=67.32
infty=np.infty
pi=np.pi
twopi=2.*pi
ln2=np.log(2)
nu_rest_21=1420.405751768 # MHz

h_Planck18=H0_Planck18/100.
Omegamh2_Planck18=Omegam_Planck18*h_Planck18**2
pars_set_cosmo_Planck18=[H0_Planck18,Omegabh2_Planck18,Omegamh2_Planck18,AS_Planck18,ns_Planck18] # suitable for get_mps

"""
"universal docstring" of args common to functions in this module
kpar           = k-parallel      modes of interest for a cylindrically binned power spectrum (assumed to be monotonic-increasing) (1/Mpc)
kperp          = k-perpendicular modes of interest for a cylindrically binned power spectrum (assumed to be monotonic-increasing) (1/Mpc)
sigLoS         = figure of merit for the width (standard deviation) of the LoS response (Mpc), modelled for now as a Gaussian
fwhmbeam       = figure of merit for the width (FWHM) of the beam response (rad), modelled for now as a Gaussian
r0             = peak comoving distance of the LoS response (Mpc)
beamtype       = Gaussian (the only one currently supported) ... in the future, might generalize to an Airy beam or something more sophisticated
save           = whether or not to save a copy of the cylindrically binned window function
savename       = distinctive tail of the name of the saved copy of the cylindrically binned window function
epsLoS         = fractional uncertainty in the width of the LoS response (positive/negative means you under/overestimate the width)
epsbeam        = fractional uncertainty in the width of the beam response (positive/negative means you under/overestimate the width)
pars_set_cosmo = fiducial cosmo params (send to CAMB, perform comoving calculations, etc.) (npset,)
pars_forecast  = cosmo params of interest for the current forecast (npfore,)
z              = redshift for which you want the cylindically binned power spectrum (probably the central redshift of a hypothetical survey)
n_sph_modes    = number of (scalar) k-modes at which the spherically binned CAMB MPS should be sampled
dpar           = vector of initial step size guesses to be used in numerical differentiation (npfore,)

args not explained here will be documented in the single function in which they appear
"""

class NvoxPracticalityError(Exception):
    pass

def get_channel_config(nu_ctr,Deltanu,evol_restriction_threshold=1./15.):
    """
    args
    nu_ctr                     = central frequency of the survey
    Deltanu                    = channel width
    evol_restriction_threshold = $N\Delta\nu/\nu ~ \Delta z/z ~$ evol_restriction_threshold (1/15 common in some HERA surveys); N = number of channels in the survey

    returns
    NDeltanu = survey bandwidth
    N        = number survey channels
    """
    NDeltanu=nu_ctr*evol_restriction_threshold
    N=NDeltanu/Deltanu
    return NDeltanu,N

def ceilflex(x,roundto=0.05):
    multceil=np.ceil(x/roundto)
    result=multceil*roundto
    print("ceilflex: x,multceil,result=",x,multceil,result)
    return result

def floorflex(x,roundto=0.005):
    multfloor=np.floor(x/roundto)
    result=multfloor*roundto
    print("floorflex: x,multfloor,result=",x,multfloor,result)
    return result

def get_padding(n):
    padding=n-1
    padding_lo=int(np.ceil(padding / 2))
    padding_hi=padding-padding_lo
    return padding_lo,padding_hi

def higher_dim_conv(P,Wcont):
    Pshape=P.shape
    Wcontshape=Wcont.shape
    if (Pshape!=Wcontshape):
        if(Pshape.T!=Wcontshape):
            assert(1==0), "window and pspec shapes must match"
        Wcont=Wcont.T # force P and Wcont to have the same shapes
    s0,s1=Pshape # by now, P and Wcont have the same shapes
    pad0lo,pad0hi=get_padding(s0)
    pad1lo,pad1hi=get_padding(s1)
    # Pp=np.pad(P,((pad0lo,pad0hi),(pad1lo,pad1hi)),"constant",constant_values=((0,0),(0,0))) # OLD OLD OLD
    # conv=convolve(Wcont,Pp,mode="valid") # NOW THAT I'M CENTRING MY WINDOW FUNCTION DIFFERENTLY, TRY THIS AGAIN BECAUSE NOW THE CORNER SLICE SHOULD WORK??! # CAN'T USE THIS ANYMORE BECAUSE IT GIVES ME THE WEIRD SIDELOBE THING/ SLICES THE MIDDLE WHEN I WANT THE CORNER (would've expected different padding or something)
    Wcontp=np.pad(Wcont,((pad0lo,pad0hi),(pad1lo,pad1hi)),"edge") # NEW NEW NEW
    conv=convolve(Wcontp,P,mode="valid")
    return conv

def W_cyl_binned(kpar,kperp,sigLoS,r0,fwhmbeam,save=False):
    """
    wrapper to multiply the LoS and flat sky approximation sky plane terms of the cylindrically binned window function, for the grid described by the k-parallel and k-perp modes of the survey of interest
    """
    nkpar=len(kpar) # try centring things to see if that mitigates edge effects at all...
    nkperp=len(kperp)
    deltakpar=kpar-kpar[nkpar//2] 
    deltakperp=kperp-kperp[nkperp//2]
    par_vec= np.exp(-deltakpar**2*sigLoS**2)
    perp_vec=np.exp(-(r0*fwhmbeam*deltakperp)**2/(2.*ln2))

    par_arr,perp_arr=np.meshgrid(par_vec,perp_vec,indexing="ij")
    meshed=par_arr*perp_arr # I really do want elementwise multiplication
    rawsum=np.sum(meshed)
    if (rawsum!=0): # normalize, but protect against division-by-zero errors
        normed=meshed/rawsum
    else:
        normed=meshed
    if (save):
        np.save('W_cyl_binned_2D_proxy'+str(time.time())+'.txt',normed)
    return normed

def calc_Wcont(kpar,kperp,sigLoS,r0,fwhmbeam,epsLoS,epsbeam): 
    """
    calculate the "contaminant" windowing amplitude that will help give rise to the so-called "contaminant power"
    """
    Wtrue=   W_cyl_binned(kpar,kperp,sigLoS,           r0,fwhmbeam            )
    Wthought=W_cyl_binned(kpar,kperp,sigLoS*(1-epsLoS),r0,fwhmbeam*(1-epsbeam)) # FOR NOW: BAKED IN THAT THE "THOUGHT" WIDTH RESPONSE PARAMS ARE UNDERESTIMATES FOR POSITIVE EPS
    Wthought=0 # FOR DIAGNOSTIC PURPOSES WHILE AVOIDING RESTRUCTURING MY CODE: CALL IT PCONT BUT REALLY HAVE IT REFLECT PTRUE
    return Wtrue-Wthought

def calc_Pcont_cyl(kpar,kperp,sigLoS,r0,fwhmbeam,pars_set_cosmo,epsLoS,epsbeam,z,n_sph_modes): 
    """
    calculate the cylindrically binned "contaminant power," following from the true and perceived window functions
    """
    print("calc_Pcont_cyl: n_sph_modes=",n_sph_modes)
    Wcont=calc_Wcont(kpar,kperp,sigLoS,r0,fwhmbeam,epsLoS,epsbeam)
    print(">> Pcont calc: calculated Wcont")
    kpargrid,kperpgrid,P=unbin_to_Pcyl(kpar,kperp,z,pars_set_cosmo=pars_set_cosmo,n_sph_modes=n_sph_modes)
    print(">> Pcont calc: unbinned CAMB pspec to cyl")
    ###
    np.save("cyl_Wcont.npy",Wcont)
    np.save("cyl_P.npy",P)
    np.save("cyl_kpargrid.npy",kpargrid)
    np.save("cyl_kperpgrid.npy",kperpgrid)
    ###
    Pcont=higher_dim_conv(P,Wcont) # new prototype, not symmetric under exchange of args: higher_dim_conv(P,Wcont)
    print(">> Pcont calc: convolved P and Wcont")
    return Pcont

def NvoxPracticalityWarning(Nvox):
    print("WARNING: the specified survey requires Nvox={:4}, which will probably lead to slow eval".format(Nvox))
    return None 

def get_L_N_for_box(kpar,kperp):
    # print("ENTERING get_L_N_for_box")
    # print("kpar[0],kperp[0],kpar[-1],kperp[-1]=",kpar[0],kperp[0],kpar[-1],kperp[-1])
    klow=np.min((kpar[0],kperp[0])) 
    # print("klow=",klow)
    khigh=np.max((kpar[-1],kperp[-1]))
    # print("khigh=",khigh)
    kmin=floorflex(klow)
    kmax=ceilflex(khigh)
    # print("kmin,kmax=",kmin,kmax)
    Lsurvbox=twopi/kmin
    Nvoxbox=int(Lsurvbox*kmax/pi)
    # print("Lsurvbox,Nvoxbox=",Lsurvbox,Nvoxbox)
    if (Nvoxbox>200):
        NvoxPracticalityWarning(Nvoxbox)
        # raise NvoxPracticalityError
    # print("EXITING get_L_N_for_box")
    return Lsurvbox,Nvoxbox

def calc_Pcont_asym(pars_set_cosmo,z,kpar,kperp,sigLoS,epsLoS,r0,beamfwhm_x,beamfwhm_y,eps_x,eps_y,Nvox=150,n_sph_modes=500,nkpar_box=15,nkperp_box=18,n_realiz=5):
    """
    calculate a cylindrically binned Pcont from an average over the power spectra formed from cylindrically-asymmetric-response-modulated brightness temp fields for a cosmological case of interest
    (you can still form a cylindrical summary statistic from brightness temp fields encoding effects beyond this symmetry)

    args
    beamfwhm_x = FWHM of the beam in one       polarization direction
    beamfwhm_y = "                 " the other "                    "
    eps_x      = fractional uncertainty in beamfwhm_x
    eps_y      = fractional uncertainty in beamfwhm_y
    Nvox   = number of voxels per side to use when constructing random realization Tb cubes

    returns
    contaminant power, calculated as the difference of subtracted spectra with config space–multiplied "true" and "thought" instrument responses
    """
    t0=time.time()
    h=pars_set_cosmo[0]/100 # typical disclaimer about cosmo param order being baked in...
    kmin_want=np.min((kpar[0],kperp[0]))           # smallest scale we care to know about (the smallest mode on one of the cyl axes)
    kmax_want=np.sqrt(kpar[-1]**2+kperp[-1]**2)    # largest scale we're interested in at any point (happens to be a spherical mode)
    ksph,Ptruesph=get_mps(pars_set_cosmo,z,minkh=kmin_want/h,maxkh=kmax_want/h,n_sph_modes=n_sph_modes)
    t1=time.time()
    print(">> Pcont calc: sourced pspec from CAMB",t1-t0)
    Lsurvbox,Nvoxbox=get_L_N_for_box(kpar,kperp)
    print("Lsurvbox,Nvoxbox=",Lsurvbox,Nvoxbox)
    nkpar=len(kpar)
    nkperp=len(kperp)
    Ptrue_realizations=   np.zeros((nkpar,nkperp,n_realiz))
    Pthought_realizations=np.zeros((nkpar,nkperp,n_realiz))
    for i in range(n_realiz):
        t2=time.time()
        # Lsurvbox,Nvoxbox=get_L_N_for_box(kpar,kperp)
        _,Tbox,rmags=generate_box(Ptruesph,ksph,Lsurvbox,Nvoxbox) 
        t3=time.time()
        print(">> Pcont calc: generated box from pspec - realization",i,t3-t2)
        if (i==0):
            t4=time.time()
            X,Y,Z=np.meshgrid(rmags,rmags,rmags,indexing="ij")
            response_true=    custom_response(X,Y,Z, sigLoS,           beamfwhm_x,          beamfwhm_y,          r0)
            response_thought= custom_response(X,Y,Z, sigLoS*(1-epsLoS),beamfwhm_x*(1-eps_x),beamfwhm_y*(1-eps_y),r0)
            t5=time.time()
            print(">> Pcont calc: generated responses",t5-t4)
        t6=time.time()
        T_x_true_resp=   Tbox* response_true
        T_x_thought_resp=Tbox* response_thought
        ###
        rgridx,rgridy=np.meshgrid(rmags,rmags,indexing="ij")
        fig,axs=plt.subplots(3,9,figsize=(22,5))
        cubes=[Tbox,response_true,T_x_true_resp]
        cubenames=["T","resp","T x resp"]
        places=[2,Nvox//2,-2]
        for ii,cube in enumerate(cubes):
            for jj,place in enumerate(places):
                # im=axs[ii,ij].pcolor(  rgridx,rgridy,cube[place,:,:])
                im=axs[ii,jj].imshow(cube[place,:,:])
                plt.colorbar(im,ax=axs[ii,jj],fraction=0.05)
                axs[ii,jj].set_title(cubenames[ii]+" "+str(place)+" axis 0")
                # im=axs[ii,j+3].pcolor(rgridx,rgridy,cube[:,place,:])
                im=axs[ii,jj+3].imshow(cube[:,place,:])
                plt.colorbar(im,ax=axs[ii,jj+3],fraction=0.05)
                axs[ii,jj+3].set_title(cubenames[ii]+" "+str(place)+" axis 1")
                # im=axs[ii,j+6].pcolor(rgridx,rgridy,cube[:,:,place])
                im=axs[ii,jj+6].imshow(cube[:,:,place])
                plt.colorbar(im,ax=axs[ii,jj+6],fraction=0.05)
                axs[ii,jj+6].set_title(cubenames[ii]+" "+str(place)+" axis 2")
        plt.suptitle("cube diagnostic plot")
        plt.tight_layout()
        plt.savefig("cube_diagnostic_plot.png",dpi=500)
        plt.show()
        # assert(1==0), "cutting off at the cube diagnostic plot for now"
        ###
        t7=time.time()
        print(">> Pcont calc: multiplied box and instrument response - realization",i,t7-t6)
        bundled_args=(sigLoS,beamfwhm_x,beamfwhm_y,r0,)
        ktrue_intrinsic_to_box,    Ptrue_intrinsic_to_box=    generate_P(T_x_true_resp,    "lin",Lsurvbox,nkpar_box,Nk1=nkperp_box, custom_estimator=custom_response,custom_estimator_args=bundled_args) # WAS LIN BUT NUMERICS WERE BAD
        # ##
        # kbox0,kbox1=ktrue_intrinsic_to_box
        # kbox0grid,kbox1grid=np.meshgrid(kbox0,kbox1,indexing="ij")
        # plt.figure()
        # plt.pcolor(kbox0grid,kbox1grid,Ptrue_intrinsic_to_box)
        # plt.title("cyl binned box-intrinsic pspec")
        # plt.savefig("box_intrinsic_pspec.png")
        # plt.show()
        # ##
        t8=time.time()
        kthought_intrinsic_to_box, Pthought_intrinsic_to_box= generate_P(T_x_thought_resp, "lin",Lsurvbox,nkpar_box,Nk1=nkperp_box, custom_estimator=custom_response,custom_estimator_args=bundled_args)
        k_survey=(kpar,kperp)
        t9=time.time()
        print(">> Pcont calc: generated pspecs from modulated boxes - realization",i,t9-t7)
        _,   Ptrue= interpolate_P(Ptrue_intrinsic_to_box,    ktrue_intrinsic_to_box,    k_survey, avoid_extrapolation=False) # the returned k are the same as the k-modes passed in k_survey
        _,Pthought= interpolate_P(Pthought_intrinsic_to_box, kthought_intrinsic_to_box, k_survey, avoid_extrapolation=False)
        t10=time.time()
        print(">> Pcont calc: re-binned pspecs to k-modes of interest - realization",i,t10-t9)
        Ptrue_realizations[:,:,i]=    Ptrue
        Pthought_realizations[:,:,i]= Pthought
        t11=time.time()
    Ptrue=    np.mean(Ptrue_realizations,    axis=-1)
    Pthought= np.mean(Pthought_realizations, axis=-1)
    Pthought=0 # FOR DIAGNOSTIC PURPOSES WHILE AVOIDING RESTRUCTURING MY CODE: CALL IT PCONT BUT REALLY HAVE IT REFLECT PTRUE
    t12=time.time()
    print(">> Pcont calc: averaged over statistical realizations to obtain Ptrue and Pthought",t12-t11)
    Pcont=Ptrue-Pthought
    t13=time.time()
    print(">> Pcont calc: subtracted Pthought from Ptrue",t13-t12)
    return Pcont

def custom_response(X,Y,Z,sigLoS,beamfwhm_x,beamfwhm_y,r0):
    """
    "custom" response function using the approximation where there is a Gaussian along the LoS and another in the sky plane

    args
    X,Y,Z      = meshgridded (indexing="ij") (Nvox,Nvox,Nvox) boxes 
    sigLoS     = characteristic width of the instrument response function along the line of sight
    beamfwhm_x = x-pol power beam fwhm
    beamfwhm_y = y-pol power beam fwhm
    r0         = central comoving distance of the survey volume 

    returns
    (Nvox,Nvox,Nvox) Cartesian box (z=LoS direction), centred at r0, sampling the response fcn at each point
    """
    return np.exp(-(Z/(2*sigLoS))**2 -ln2*((X/beamfwhm_x)**2+(Y/beamfwhm_y)**2)/r0**2) # LoS direction is z-like! everything is compatible here! see the comments in the elif (binto=="cyl") branch of P_driver in power.py to remind myself!!
    # if (np.any(np.array([sigLoS,beamfwhm_x,beamfwhm_y]))==0):
    #     return 0.*X
    # else:
    #     return np.exp(-(Z/(2*sigLoS))**2 -ln2*((X/beamfwhm_x)**2+(Y/beamfwhm_y)**2)/r0**2) # LoS direction is z-like! everything is compatible here! see the comments in the elif (binto=="cyl") branch of P_driver in power.py to remind myself!!

def unbin_to_Pcyl(kpar,kperp,z,pars_set_cosmo=pars_set_cosmo_Planck18,n_sph_modes=500):  
    """
    interpolate a spherically binned CAMB MPS to provide MPS values for a cylindrically binned k-grid of interest (nkpar x nkperp)
    """
    h=pars_set_cosmo[0]/100.
    kmin=np.sqrt(kpar[0]**2+kperp[0]**2)
    kmax=np.sqrt(kpar[-1]**2+kperp[-1]**2)
    k,Psph=get_mps(pars_set_cosmo,z,minkh=kmin/h,maxkh=kmax/h,n_sph_modes=n_sph_modes)
    Psph=Psph.reshape((Psph.shape[1],))
    kpargrid,kperpgrid=np.meshgrid(kpar,kperp,indexing="ij")
    Pcyl=np.zeros((len(kpar),len(kperp)))
    for i,kpar_val in enumerate(kpar):
        for j,kperp_val in enumerate(kperp):
            k_of_interest=np.sqrt(kpar_val**2+kperp_val**2)
            idx_closest_k=np.argmin(np.abs(k-k_of_interest)) # k-scalar in the CAMB MPS closest to the k-magnitude indicated by the kpar-kperp combination for that point in cylindrically binned Fourier space
            if (idx_closest_k==0): # start of array
                idx_2nd_closest_k=1 # use hi
            elif (idx_closest_k==n_sph_modes-1): # end of array
                idx_2nd_closest_k=n_sph_modes-2 # use lo
            else: # middle of array -> check if hi or lo is closer
                k_neighb_lo=k[idx_closest_k-1]
                k_neighb_hi=k[idx_closest_k+1]
                if (np.abs(k_neighb_lo-k_of_interest)<np.abs(k_neighb_hi-k_of_interest)): # use k_neighb_lo
                    idx_2nd_closest_k=idx_closest_k-1
                else:
                    idx_2nd_closest_k=idx_closest_k+1
            k_closest=k[idx_closest_k]
            k_2nd_closest=k[idx_2nd_closest_k]
            interp_slope=(Psph[idx_2nd_closest_k]-Psph[idx_closest_k])/(k_2nd_closest-k_closest)
            Pcyl[i,j]=interp_slope*(k_of_interest-k_closest)
    return kpargrid,kperpgrid,Pcyl

scale=1e-9
def get_mps(pars_set_cosmo,z,minkh=1e-4,maxkh=1,n_sph_modes=500):
    """
    get matter power spectrum from CAMB

    args
    minkh = min value of k/h at which to calculate the MPS
    maxkh = max value of k/h at which to calculate the MPS 
    """
    z=[z]
    H0=pars_set_cosmo[0]
    h=H0/100.
    ombh2=pars_set_cosmo[1]
    omch2=pars_set_cosmo[2]
    As=pars_set_cosmo[3]*scale
    ns=pars_set_cosmo[4]

    pars_set_cosmo=camb.set_params(H0=H0, ombh2=ombh2, omch2=omch2, ns=ns, mnu=0.06,omk=0)
    # camb.dark_energy.DarkEnergyEqnOfState.set_params(w=, wa=)
    pars_set_cosmo.InitPower.set_params(As=As,ns=ns,r=0)
    pars_set_cosmo.set_matter_power(redshifts=z, kmax=maxkh*h)
    results = camb.get_results(pars_set_cosmo)
    pars_set_cosmo.NonLinear = model.NonLinear_none
    kh,z,pk=results.get_matter_power_spectrum(minkh=minkh,maxkh=maxkh,npoints=n_sph_modes)
    return kh,pk

def cyl_partial(pars_set_cosmo,z,n,dpar,kpar,kperp,n_sph_modes=500,ftol=1e-6,eps=1e-16,maxiter=5):  
    """
    args
    n       = take the partial derivative WRT the nth parameter in p
    ftol    = fractional tolerance relating to the scale of the function (defined for points of interest)
    eps     = tiny offset factor to protect against numerical division-by-zero errors
    maxiter = maximum number of times to let the step size optimization attempt recurse before "giving up" and using the most recent guess

    returns
    cylindrically binned matter power spectrum partial WRT one cosmo parameter (nkpar x nkperp)
    """
    done=False
    iter=0
    dparn=dpar[n]
    pcopy=pars_set_cosmo.copy()
    pndispersed=pcopy[n]+np.linspace(-2,2,5)*dparn

    pcopy=pars_set_cosmo.copy()
    _,_,Pcyl_base=unbin_to_Pcyl(kpar,kperp,z,pars_set_cosmo=pcopy,n_sph_modes=n_sph_modes)
    P0=np.mean(np.abs(Pcyl_base))+eps
    tol=ftol*P0 # generalizes tol=ftol*f0 from 512

    pcopy[n]=pcopy[n]+2*dparn
    _,_,Pcyl_2plus=unbin_to_Pcyl(kpar,kperp,z,pars_set_cosmo=pcopy,n_sph_modes=n_sph_modes)
    pcopy=pars_set_cosmo.copy()
    pcopy[n]=pcopy[n]-2*dparn
    _,_,Pcyl_2minu=unbin_to_Pcyl(kpar,kperp,z,pars_set_cosmo=pcopy,n_sph_modes=n_sph_modes)
    deriv1=(Pcyl_2plus-Pcyl_2minu)/(4*dpar[n])

    pcopy[n]=pcopy[n]+dparn
    _,_,Pcyl_plus=unbin_to_Pcyl(kpar,kperp,z,pars_set_cosmo=pcopy,n_sph_modes=n_sph_modes)
    pcopy=pars_set_cosmo.copy()
    pcopy[n]=pcopy[n]-dparn
    _,_,Pcyl_minu=unbin_to_Pcyl(kpar,kperp,z,pars_set_cosmo=pcopy,n_sph_modes=n_sph_modes)
    deriv2=(Pcyl_plus-Pcyl_minu)/(2*dpar[n])

    while (done==False):
        if (np.any(Pcyl_plus-Pcyl_minu)<tol): # consider relaxing this to np.any if it ever seems like too strict a condition?!
            estimate=(4*deriv2-deriv1)/3
            return estimate # higher-order estimate
        else:
            pnmean=np.mean(np.abs(pndispersed)) # the np.abs part should be redundant because, by this point, all the k-mode values and their corresponding dpns and Ps should be nonnegative, but anyway... numerical stability or something idk
            Psecond=np.abs(2*Pcyl_base-Pcyl_minu-Pcyl_plus)/dpar[n]**2
            dparn=np.sqrt(eps*pnmean*P0/Psecond)
            iter+=1
            if iter==maxiter:
                print("failed to converge in {:d} iterations".format(maxiter))
                fallback=(4*deriv2-deriv1)/3
                print("RETURNING fallback")
                return fallback

def build_cyl_partials(pars_set_cosmo,z,n_sph_modes,kpar,kperp,dpar):
    """
    builds a (npfore,nkpar,nkperp) array of the partials of the cylindrically binned MPS WRT each cosmo param in the forecast
    """
    nkpar=len(kpar)
    nkperp=len(kperp)
    nprm=len(pars_set_cosmo)
    V=np.zeros((nprm,nkpar,nkperp))
    for n in range(nprm):
        V[n,:,:]=cyl_partial(pars_set_cosmo,z,n,dpar,kpar,kperp,n_sph_modes=n_sph_modes)
    return V

def bias(partials,unc, kpar,kperp,sigLoS,r0,fwhmbeam0,pars_set_cosmo,epsLoS,epsbeam0,z,n_sph_modes,savename=None,cyl_sym_resp=True, fwhmbeam1=1e-3,epsbeam1=0.1,Nvox=150,recalc_Pcont=False,n_realiz=5):
    """
    args
    partials = npfore x nkpar x nkperp array where each slice of constant 0th (nprm) index is an nkpar x nkperp array of the MPS's partial WRT a particular parameter in the forecast
    unc      = nkpar x nkperp array describing the standard deviations at each cylindrically binned k-mode

    returns
    (npfore,) vector of biases resulting from beam mismodelling for the parameters of interest in the forecast
    """
    V=0.0*partials # still want the same shape as the vector of partials, even though this is different than for the spherical case
    nprm=partials.shape[0]
    uncsh0,uncsh1=unc.shape
    partsh0,partsh1,partsh2=partials.shape
    if (uncsh0==partsh2 and uncsh1==partsh1):
        unc=unc.T

    for i in range(nprm):
        V[i,:,:]=partials[i,:,:]/unc # elementwise division for an nkpar x nkperp slice
    V_completely_transposed=np.transpose(V,axes=(2,1,0)) # from the docs: "For an n-D array, if axes are given, their order indicates how the axes are permuted"
    F=np.einsum("ijk,kjl->il",V,V_completely_transposed)
    print("computed F")
    if recalc_Pcont:
        if cyl_sym_resp:
            Pcont=calc_Pcont_cyl(kpar,kperp,sigLoS,r0,fwhmbeam0,pars_set_cosmo,epsLoS,epsbeam0,z,n_sph_modes)
        else:
            Pcont=calc_Pcont_asym(pars_set_cosmo,z,
                                  kpar,kperp,
                                  sigLoS,epsLoS,r0,fwhmbeam0,fwhmbeam1,epsbeam0,epsbeam1,
                                  Nvox=Nvox,n_sph_modes=n_sph_modes,n_realiz=n_realiz) 
    else:
        Pcont=np.load(savename)

    print("computed Pcont")
    np.save("Pcont_"+savename+".npy",Pcont)
    Pcont_div_sigma=Pcont/unc
    B=np.einsum("jk,ijk->i",Pcont_div_sigma,V)
    print("computed B")
    bias=(np.linalg.inv(F)@B).reshape((F.shape[0],))
    print("computed b")
    return bias

def printparswbiases(pars_set_cosmo,parnames,biases):
    """
    args
    parnames = (npfore,) vector of strings: names of the parameters in the forecast (assumed to be in the same order as pars_set_cosmo)
    biases   = (npfore,) vector: biases in estimating the cosmo params in the forecast resulting from inadvertent beam mismodelling (in the same order as pars_set_cosmo and parnames)

    returns
    None (fcn prints a formatted summary of the forecasting pipeline calculation)
    """
    print("\n\nbias calculation results for the survey described above.................................")
    print("........................................................................................")
    for p,par in enumerate(pars_set_cosmo):
        print('{:12} = {:-10.3e} with bias {:-12.5e} (fraction = {:-10.3e})'.format(parnames[p], par, biases[p], biases[p]/par))
    return None