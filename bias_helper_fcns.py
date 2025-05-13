import numpy as np
# from matplotlib import pyplot as plt
# import pygtc
import camb
from camb import model
# from numpy.fft import rfft2,irfft2
from scipy.signal import convolve
# from cyl_bin_window import *

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
c=2.998e8 # m s^{-1}
pc=30856775814914000 # m
Mpc=pc*1e6

h_Planck18=H0_Planck18/100.
Omegamh2_Planck18=Omegam_Planck18*h_Planck18**2
pars_Planck18=[H0_Planck18,Omegabh2_Planck18,Omegamh2_Planck18,AS_Planck18,ns_Planck18] # suitable for get_mps

def higher_dim_conv(ff,gg):
    f=np.asarray(ff)
    g=np.asarray(gg)
    if (f.shape!=g.shape):
        if(f.shape!=(g.T).shape):
            assert(1==0), "incompatible array shapes"
        else: # need to transpose g for things to work as expected
            g=g.T
    a,b=f.shape # if eval makes it to this point, f.shape==g.shape is True

    fp=np.zeros((2*a,2*b)) # holders for padded versions (wraparound-safe...)
    gp=np.zeros((2*a,2*b))
    fp[:a,:b]=f # populate the zero-padded versions
    gp[:a,:b]=g

    result_p=convolve(fp,gp)
    result=result_p[:a,:b]
    return result

def calc_par_vec(deltakparvec,sigLoS,r0):
    expterm=twopi*sigLoS**2*np.exp(-deltakparvec**2*sigLoS**2)
    return expterm

def calc_perp_vec(deltakperpvec,Dc,sigbeam,beamtype="Gaussian"):
    beamtype=beamtype.lower()
    if beamtype=="gaussian":
        alpha=ln2/(sigbeam*Dc)**2
        # print("Gaussian alpha=ln2/(sigbeam*Dc)**2=",alpha)
        vec=np.exp(-deltakperpvec**2/(2*alpha))
    # elif beamtype=="airy":
    #     nk=len(deltakperpvec)
    #     vec=np.zeros(nk)
    #     alpha=sigbeam_to_alpha(sigbeam)
    #     print("Airy alpha=sigbeam_to_alpha(sigbeam)=",alpha)
    #     vec=airy(deltakperpvec,alpha)
    else: 
        assert(1==0), "only a Gaussian beam is currently supported"
        # assert(1==0), "currently supported beam types are Airy and Gaussian"
    return vec

# def W_cyl_binned(deltakparvec,deltakperpvec,sigLoS,r0,sigbeam,save=False,savename="test",btype="Gaussian",plot=False): 
#     par_vec=calc_par_vec(deltakparvec,sigLoS,r0)
#     perp_vec=calc_perp_vec(deltakperpvec,r0,sigbeam,beamtype=btype)
#     par_arr,perp_arr=np.meshgrid(par_vec,perp_vec)
#     meshed=par_arr*perp_arr # interested in elementwise (not matrix) multiplication
#     if (save):
#         np.save('W_cyl_binned_2D_proxy'+str(time.time())+'.txt',normed)
#     return meshed # NO LONGER NORMALIZING BC HANDLING AT THE WCONT LEVEL

# def calc_Wcont(kpar,kperp,sigLoS,r0,sigbeam,epsLoS,epsbeam,savestat="False",saven=None,beamtype="Gaussian"):
#     kpargrid,kperpgrid=np.meshgrid(kpar,kperp)
#     Wbase=W_cyl_binned(kpar,kperp,sigLoS,r0,sigbeam,save=savestat,savename=saven,btype=beamtype)
#     Wcont_shape=Wbase*np.sqrt((kpargrid**2*sigLoS*epsLoS)**2+(kperpgrid**2*sigbeam*epsbeam)**2)
#     rawsum=np.sum(Wcont_shape)
#     if (rawsum!=0): 
#         return Wcont_shape/rawsum
#     else:
#         return meshed

def W_cyl_binned(deltakparvec,deltakperpvec,sigLoS,r0,sigbeam,save=False,savename="test",btype="Gaussian",plot=False): 
    # print("in W_cyl_binned:\nlen(kpar),len(kperp)=",len(deltakparvec),len(deltakperpvec))
    par_vec=calc_par_vec(deltakparvec,sigLoS,r0)
    perp_vec=calc_perp_vec(deltakperpvec,r0,sigbeam,beamtype=btype)
    par_arr,perp_arr=np.meshgrid(par_vec,perp_vec,indexing="ij")
    # print("in W_cyl_binned: par_arr.shape,perp_arr.shape=",par_arr.shape,perp_arr.shape)
    meshed=par_arr*perp_arr # interested in elementwise (not matrix) multiplication
    # print("in W_cyl_binned:\nmeshed.shape=",meshed.shape)
    rawsum=np.sum(meshed)
    if (rawsum!=0):
        normed=meshed/rawsum
    else:
        normed=meshed
    if (save):
        np.save('W_cyl_binned_2D_proxy'+str(time.time())+'.txt',normed)
    # print("in W_cyl_binned:\nnormed.shape=",normed.shape)
    return normed # RETURN TO HANDLING NORMALIZATION AT THIS LEVEL

def calc_Wcont(kpar,kperp,sigLoS,r0,sigbeam,epsLoS,epsbeam,savestat="False",saven=None,beamtype="Gaussian"):
    Wtrue=   W_cyl_binned(kpar,kperp,sigLoS,           r0,sigbeam,            save=savestat,savename=saven,btype=beamtype)
    Wthought=W_cyl_binned(kpar,kperp,sigLoS*(1-epsLoS),r0,sigbeam*(1-epsbeam),save=savestat,savename=saven,btype=beamtype) # FOR NOW: BAKED IN THAT THE "THOUGHT" WIDTH RESPONSE PARAMS ARE UNDERESTIMATES FOR POSITIVE EPS
    return Wtrue-Wthought

def calc_Pcont_cyl(kpar,kperp,sigLoS,r0,sigbeam,pars,epsLoS,epsbeam,z,n_sph_pts,beamtype="Gaussian",savestatus=False,savename=None): # V4
    Wcont=calc_Wcont(kpar,kperp,sigLoS,r0,sigbeam,epsLoS,epsbeam,savestat=savestatus,saven=savename,beamtype=beamtype)
    kpargrid,kperpgrid,P=unbin_to_Pcyl(kpar,kperp,z,pars=pars,nsphpts=n_sph_pts)
    print("in calc_Pcont_cyl:\n Wcont.shape,Ptrue.shape=",Wcont.shape,P.shape)
    Pcont=higher_dim_conv(Wcont,P)
    return Pcont

def unbin_to_Pcyl(kpar,kperp,z,pars=pars_Planck18,nsphpts=500):
    """
    kpar    = k-parallel modes of interest for a cylindrically binned power spectrum emulation (assumed to be monotonic-increasing)
    kperp   = k-perp modes of interest for a cylindrically binned power spectrum emulation (assumed to be monotonic-increasing)
    z       = redshift for which you want the cylindically binned power spectrum
    pars    = cosmo params to use to generate a spherically binned MPS in CAMB
    nsphpts = number of (scalar) k-modes at which the spherically binned CAMB MPS should be sampled
    """
    h=pars[0]/100.
    kmin=np.sqrt(kpar[0]**2+kperp[0]**2)
    kmax=np.sqrt(kpar[-1]**2+kperp[-1]**2)
    k,Psph=get_mps(pars,z,minkh=kmin/h,maxkh=kmax/h,npts=nsphpts) # get_mps(pars,zs,minkh=1e-4,maxkh=1,npts=200)
    Psph=Psph.reshape((Psph.shape[1],))
    kpargrid,kperpgrid=np.meshgrid(kpar,kperp,indexing="ij")
    Pcyl=np.zeros((len(kpar),len(kperp)))
    for i,kpar_val in enumerate(kpar):
        for j,kperp_val in enumerate(kperp):
            k_of_interest=np.sqrt(kpar_val**2+kperp_val**2)
            idx_closest_k=np.argmin(np.abs(k-k_of_interest)) # k-scalar in the CAMB MPS closest to the k-magnitude indicated by the kpar-kperp combination for that point in cylindrically binned Fourier space
            if (idx_closest_k==0): # start of array
                idx_2nd_closest_k=1 # use hi
            elif (idx_closest_k==nsphpts-1): # end of array
                idx_2nd_closest_k=nsphpts-2 # use lo
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
def get_mps(pars,zs,minkh=1e-4,maxkh=1,npts=500):
    '''
    get matter power spectrum

    pars   = vector of cosmological parameters (npar x 1)
    zs     = redshifts of interest (**tuple** of floats)
    kmax   = max wavenumber to calculate the MPS for
    linear = if True, calc linear matter PS; else calc NL MPS (Boolean)
    minkh  = min value of k/h to calculate the MPS for
    maxkh  = max value of k/h to calculate the MPS for
    npts   = number of points in the calculated MPS
    '''
    zs=[zs]
    H0=pars[0]
    h=H0/100.
    ombh2=pars[1]
    omch2=pars[2]
    As=pars[3]*scale
    ns=pars[4]

    pars=camb.set_params(H0=H0, ombh2=ombh2, omch2=omch2, ns=ns, mnu=0.06,omk=0)
    pars.InitPower.set_params(As=As,ns=ns,r=0)
    pars.set_matter_power(redshifts=zs, kmax=maxkh*h)
    lin=True
    results = camb.get_results(pars)
    if lin:
        pars.NonLinear = model.NonLinear_none
    else:
        pars.NonLinear = model.NonLinear_both

    kh,z,pk=results.get_matter_power_spectrum(minkh=minkh,maxkh=maxkh,npoints=npts)
    return kh,pk

def cyl_partial(p,zs,n,dpar,kpar,kperp,nmodes_sph=200,ftol=1e-6,eps=1e-16,maxiter=5): # get_mps(pars,zs,minkh=1e-4,maxkh=1,npts=500)
    '''
    args
    p           = vector of cosmological parameters (npar x 1)
    zs          = tuple of redshifts where we're interested in calculating the MPS
    n           = take the partial derivative WRT the nth parameter in p
    dpar        = vector of step sizes (npar x 1)
    kpar, kperp = cylindrically binned k-modes where you're interested in commenting on the MPS
    nmodes_sph  = number of k-modes at which to sample the CAMB (spherically binned) MPS

    returns
    cylindrically binned matter power spectrum partial WRT one cosmo parameter (nkpar x nkperp)
    '''
    done=False
    iter=0
    nkpar=len(kpar)
    nkperp=len(kperp)
    dparn=dpar[n]
    pcopy=p.copy()
    pndispersed=pcopy[n]+np.linspace(-2,2,5)*dparn

    pcopy=p.copy()
    _,_,Pcyl_base=unbin_to_Pcyl(kpar,kperp,zs,pars=pcopy,nsphpts=nmodes_sph)
    P0=np.mean(np.abs(Pcyl_base))+eps
    tol=ftol*P0 # generalizes tol=ftol*f0 from 512

    pcopy[n]=pcopy[n]+2*dparn
    _,_,Pcyl_2plus=unbin_to_Pcyl(kpar,kperp,zs,pars=pcopy,nsphpts=nmodes_sph)
    pcopy=p.copy()
    pcopy[n]=pcopy[n]-2*dparn
    _,_,Pcyl_2minu=unbin_to_Pcyl(kpar,kperp,zs,pars=pcopy,nsphpts=nmodes_sph)
    deriv1=(Pcyl_2plus-Pcyl_2minu)/(4*dpar[n])

    pcopy[n]=pcopy[n]+dparn
    _,_,Pcyl_plus=unbin_to_Pcyl(kpar,kperp,zs,pars=pcopy,nsphpts=nmodes_sph)
    pcopy=p.copy()
    pcopy[n]=pcopy[n]-dparn
    _,_,Pcyl_minu=unbin_to_Pcyl(kpar,kperp,zs,pars=pcopy,nsphpts=nmodes_sph)
    deriv2=(Pcyl_plus-Pcyl_minu)/(2*dpar[n])

    if (np.all(Pcyl_plus-Pcyl_minu)<tol): # consider relaxing this to np.any if it ever seems like too strict a condition?!
        print("current step size okay -> returning a derivative estimate")
        return (4*deriv2-deriv1)/3 # higher-order estimate
    else:
        pnmean=np.mean(np.abs(pndispersed)) # the np.abs part should be redundant because, by this point, all the k-mode values and their corresponding dpns and Ps should be nonnegative, but anyway... numerical stability or something idk
        Psecond=np.abs(2*Pcyl_base-Pcyl_minu-Pcyl_plus)/dx**2
        dparn=np.sqrt(eps*pnmean*P0/Psecond)
        iter+=1
        if iter==maxiter:
            print("failed to converge in {:d} iterations".format(maxiter))
            return (4*deriv2-deriv1)/3

def build_cyl_partials(p,z,nmodes_sph,kpar,kperp,dpar):
    nkpar=len(kpar)
    nkperp=len(kperp)
    nprm=len(p)
    V=np.zeros((nprm,nkpar,nkperp))
    for n in range(nprm):
        V[n,:,:]=cyl_partial(p,z,n,dpar,kpar,kperp,nmodes_sph=nmodes_sph)
    return V

def bias(partials,unc, kpar,kperp,sigLoS,r0,sigbeam,pars,epsLoS,epsbeam,z,n_sph_pts,beamtype="Gaussian",savestatus=False,savename=None): # new 13/05/25
    '''
    partials = nprm x nkpar x nkperp array where each slice of constant 0th (nprm) index is an nkpar x nkperp array of the MPS's partial WRT a particular parameter in the forecast
    unc      = nnpar x nkperp array describing the standard deviations at each cylindrically binned k-mode
    '''
    V=0.0*partials # still want the same shape as the vector of partials, even though this is different than for the spherical case
    nprm=partials.shape[0]
    uncsh0,uncsh1=unc.shape
    partsh0,partsh1,partsh2=partials.shape
    print("partials.shape,unc.shape=",partials.shape,unc.shape)
    if (uncsh0==partsh2 and uncsh1==partsh1):
        print("transposing unc")
        unc=unc.T

    for i in range(nprm):
        V[i,:,:]=partials[i,:,:]/unc # elementwise division for a nkpar x nkperp slice
    print("V.shape=",V.shape)
    V_completely_transposed=np.transpose(V,axes=(2,1,0)) # from the docs: "For an n-D array, if axes are given, their order indicates how the axes are permuted"
    print("V_completely_transposed.shape=",V_completely_transposed.shape)
    F=np.einsum("ijk,kjl->il",V,V_completely_transposed)
    print("F.shape=",F.shape)
    Pcont=calc_Pcont_cyl(kpar,kperp,sigLoS,r0,sigbeam,pars,epsLoS,epsbeam,z,n_sph_pts,savestatus=savestatus,savename=savename,beamtype=beamtype)
    print("Pcont.shape,unc.shape=",Pcont.shape,unc.shape)
    Pcont_div_sigma=Pcont/unc
    print("Pcont_div_sigma.shape=",Pcont_div_sigma.shape)
    # B=np.einsum("ij,ijk->k",Pcont_div_sigma,V_completely_transposed)
    B=np.einsum("jk,ijk->i",Pcont_div_sigma,V)
    print("B.shape=",B.shape)
    bias=(np.linalg.inv(F)@B).reshape((F.shape[0],))
    print("bias.shape=",bias.shape)
    return bias

def printparswbiases(pars,parnames,biases):
    for i in range(5):
        print(".")
    for p,par in enumerate(pars):
        print('{:12} = {:-10.3e} with bias {:-12.5e}'.format(parnames[p], par, biases[p]))
    return None