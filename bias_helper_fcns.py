import numpy as np
# from matplotlib import pyplot as plt
# import pygtc
import camb
from camb import model
from numpy.fft import rfft2,irfft2
from cyl_bin_window import *

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

    Fp=rfft2(fp)
    Gp=rfft2(gp)
    Fourier_space_product_p=Fp*Gp # _p for padded
    result_p=irfft2(Fourier_space_product_p)
    result=result_p[:a,:b]
    return result

def calc_Pcont_cyl(kpar,kperp,sigLoS,r0,thetaHWHM,savestatus,savename,beamtype,pars,eps,z,n_sph_pts):
    W=        W_cyl_binned(kpar,kperp,sigLoS,r0,        thetaHWHM,save=savestatus,savename=savename,btype=beamtype)
    Wthought= W_cyl_binned(kpar,kperp,sigLoS,r0,(1+eps)*thetaHWHM,save=savestatus,savename=savename,btype=beamtype) # modify if I want to study a different mischaracterization!
    deltaW=W-Wthought
    kpargrid,kperpgrid,P=unbin_to_Pcyl(kpar,kperp,z,pars=pars,nsphpts=n_sph_pts)
    # print("tracking down shape errors in calc_Pcont_cyl:")
    # print("deltaW.shape=",deltaW.shape)
    # print("P.shape=",P.shape)
    # print("end of calc_Pcont_cyl print checks")
    Pcont=higher_dim_conv(deltaW,P)
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
    kpargrid,kperpgrid=np.meshgrid(kpar,kperp)
    Pcyl=np.zeros((len(kpar),len(kperp)))
    for i,kpar_val in enumerate(kpar):
        for j,kperp_val in enumerate(kperp):
            k_of_interest=np.sqrt(kpar_val**2+kperp_val**2)
            idx_closest_k=np.argmin(np.abs(k-k_of_interest)) # k-scalar in the CAMB MPS closest to the k-magnitude indicated by the kpar-kperp combination for that point in cylindrically binned Fourier space
            Pcyl[i,j]=Psph[idx_closest_k]
    return kpargrid,kperpgrid,Pcyl

scale=1e-9
def get_mps(pars,zs,minkh=1e-4,maxkh=1,npts=500): # < CAMBpartial < buildCAMBpartials
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

def CAMBpartial(p,zs,n,dpar,nmodes=200): # < buildCAMBpartial
    '''
    args
    p    = vector of cosmological parameters (npar x 1)
    zs   = tuple of redshifts where we're interested in calculating the MPS
    n    = take the partial derivative WRT the nth parameter in p
    dpar = vector (you might want dif step sizes for dif params) of step sizes (npar x 1)

    returns
    spherically binned matter power spectrum partial WRT one cosmo parameter (nk x 1)
    '''
    kh,pk=get_mps(p,zs,npts=nmodes)
    npts=pk.shape[1]
    pcopy=p.copy()
    pcopy[n]=pcopy[n]+dpar[n]
    khp,pkp=get_mps(pcopy,zs,npts=nmodes)
    fplus=np.array(pkp[:npts])
    pcopy=p.copy()
    pcopy[n]=pcopy[n]-dpar[n]
    khm,pkm=get_mps(pcopy,zs,npts=nmodes)
    fminu=np.array(pkm[:npts])
    return ((fplus-fminu)/(2*dpar[n])).reshape((npts,))

def cyl_partial(p,zs,n,dpar,kpar,kperp,nmodes_sph=200):
    '''
    args
    p           = vector of cosmological parameters (npar x 1)
    zs          = tuple of redshifts where we're interested in calculating the MPS
    n           = take the partial derivative WRT the nth parameter in p
    dpar        = vector of step sizes (npar x 1)
    kpar, kperp = cylindrically binned k-modes where you're interested in commenting on the MPS
    nmodes_sph  = number of k-modes at which to sample the CAMB (spherically binned) MPS

    returns
    reverse-engineered (hackily emulated) cylindrically binned matter power spectrum partial WRT one cosmo parameter (nkpar x nkperp)
    '''
    nkpar=len(kpar)
    nkperp=len(kperp)
    pcopy=p.copy()
    pcopy[n]=pcopy[n]+dpar[n]
    _,_,Pcyl_plus=unbin_to_Pcyl(kpar,kperp,zs,pars=pcopy,nsphpts=nmodes_sph)
    # print("Pcyl_plus.shape=",Pcyl_plus.shape)
    pcopy=p.copy()
    pcopy[n]=pcopy[n]-dpar[n]
    _,_,Pcyl_minu=unbin_to_Pcyl(kpar,kperp,zs,pars=pcopy,nsphpts=nmodes_sph)
    # print("Pcyl_minu.shape=",Pcyl_minu.shape)
    deriv=(Pcyl_plus-Pcyl_minu)/(2*dpar[n])
    # print("deriv.shape=",deriv.shape)
    return deriv

def buildCAMBpartials(p,z,NMODES,dpar): # output to fisher
    '''
    m      = vector of modes you want to sample your power spectrum at (nmodes x 1)
    p      = vector of cosmological parameters (npar x 1)
    dpar   = vector (since you might want dif step sizes for dif params) of step sizes (npar x 1)
    nmode = [scalar] number of modes in the spectrum - could be l-modes for CMB, k-modes for 21 cm, etc.
    '''
    nprm=len(p)
    V=np.zeros((NMODES,nprm))
    for n in range(nprm):
        V[:,n]=CAMBpartial(p,z,n,dpar,nmodes=NMODES) # THIS CALL IS WRONG?? ... for CAMB, I call build_partials with getP=CAMBpartial, which is called as CAMBpartial(p,zs,n,dpar)
    return V

def build_cyl_partials(p,z,nmodes_sph,kpar,kperp,dpar):
    nkpar=len(kpar)
    nkperp=len(kperp)
    nprm=len(p)
    V=np.zeros((nprm,nkpar,nkperp))
    for n in range(nprm):
        V[n,:,:]=cyl_partial(p,z,n,dpar,kpar,kperp,nmodes_sph=nmodes_sph)
    return V

def fisher(partials,unc): # output to cornerplot or bias
    '''
    partials = nmodes x nprm array where each column is an nmodes x 1 vector of the PS's partial WRT a particular parameter in the forecast
    unc      = nmodes x 1 vector of standard deviations at each mode (could be k-mode, l-mode, etc.)
    '''
    V=0.0*partials # want the same shape
    nprm=partials.shape[1]
    for i in range(nprm):
        V[:,i]=partials[:,i]/unc
    return V.T@V

def fisher_and_B_cyl(partials,unc, kpar,kperp,sigLoS,r0,thetaHWHM,savestatus,savename,beamtype,pars,eps,z,n_sph_pts): # B is no longer something I basically get for free and can build trivially using mat mult, so might as well use the building blocks that I already need to construct inside this function layer
    '''
    partials = nprm x nkpar x nkperp array where each slice of constant 0th (nprm) index is an nkpar x nkperp array of the MPS's partial WRT a particular parameter in the forecast
    unc      = nnpar x nkperp array describing the standard deviations at each cylindrically binned k-mode
    '''
    ## calc_Pcont_cyl(kpar,kperp,sigLoS,r0,thetaHWHM,savestatus,savename,beamtype,pars,eps,z,n_sph_pts)
    V=0.0*partials # still want the same shape, even though it is different than for the spherical case
    nprm=partials.shape[0]
    # print("start of fisher_and_B_cyl shape checks")
    # print("before shape compatibility modifications:")
    # print("partials.shape=",partials.shape)
    # print("unc.shape=",unc.shape)
    uncsh0,uncsh1=unc.shape
    partsh0,partsh1,partsh2=partials.shape
    if (uncsh0==partsh2 and uncsh1==partsh1):
        uncT=unc.T
    # print("after shape compatibility modifications:")
    # print("partials.shape=",partials.shape)
    # print("unc.shape=",unc.shape)

    for i in range(nprm):
        V[i,:,:]=partials[i,:,:]/uncT # elementwise division for a nkpar x nkperp slice
    V_completely_transposed=np.transpose(V,axes=(2,1,0)) # from the docs: "For an n-D array, if axes are given, their order indicates how the axes are permuted"
    F=np.einsum("ijk,kjl->il",V,V_completely_transposed)
    Pcont=calc_Pcont_cyl(kpar,kperp,sigLoS,r0,thetaHWHM,savestatus,savename,beamtype,pars,eps,z,n_sph_pts)
    Pcont_div_sigma=Pcont/unc
    B=np.einsum("ij,ijk->k",Pcont_div_sigma,V_completely_transposed)
    # print("end of fisher_and_B_cyl shape checks")
    return F,B

def bias(F,B):
    return (np.linalg.inv(F)@B).reshape((F.shape[0],))

def printparswbiases(pars,parnames,biases):
    for p,par in enumerate(pars):
        print('{:12} = {:-10.3e} with bias {:-12.5e}'.format(parnames[p], par, biases[p]))
    return None
