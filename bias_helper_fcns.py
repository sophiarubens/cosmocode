import numpy as np
import camb
from camb import model
from scipy.signal import convolve
from power import *

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
pars_Planck18=[H0_Planck18,Omegabh2_Planck18,Omegamh2_Planck18,AS_Planck18,ns_Planck18] # suitable for get_mps

"""
"universal docstring" of args common to functions in this module
kpar        = k-parallel      modes of interest for a cylindrically binned power spectrum (assumed to be monotonic-increasing) (1/Mpc)
kperp       = k-perpendicular modes of interest for a cylindrically binned power spectrum (assumed to be monotonic-increasing) (1/Mpc)
sigLoS      = figure of merit for the width (standard deviation) of the LoS response (Mpc), modelled for now as a Gaussian
fwhmbeam    = figure of merit for the width (FWHM) of the beam response (rad), modelled for now as a Gaussian
r0          = peak comoving distance of the LoS response (Mpc)
beamtype    = Gaussian (the only one currently supported) ... in the future, might generalize to an Airy beam or something more sophisticated
save        = whether or not to save a copy of the cylindrically binned window function
savename    = distinctive tail of the name of the saved copy of the cylindrically binned window function
epsLoS      = fractional uncertainty in the width of the LoS response (positive/negative means you under/overestimate the width)
epsbeam     = fractional uncertainty in the width of the beam response (positive/negative means you under/overestimate the width)
pars        = cosmo params of interest for the current forecast (ncp,)
z           = redshift for which you want the cylindically binned power spectrum (probably the central redshift of a hypothetical survey)
n_sph_modes = number of (scalar) k-modes at which the spherically binned CAMB MPS should be sampled
dpar        = vector of initial step size guesses to be used in numerical differentiation (ncp,)

args not explained here will be documented in the single function in which they appear
"""

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

def higher_dim_conv(ff,gg):
    """
    args
    ff, gg = two 2D arrays you want to convolve (should have the same shape)

    returns
    wraparound-protected scipy.signal.convolve(ff[a-c,b-d],gg[a,b]) conditioned to mimic the tensor multiplication np.einsum(ff[a,b,c,d],gg[a,b]) **intermediate math-code notation for clarity
    """
    f=np.asarray(ff)
    g=np.asarray(gg)
    if (f.shape!=g.shape):
        if(f.shape!=(g.T).shape):
            assert(1==0), "incompatible array shapes"
        else: # need to transpose g for things to work as expected (fcn relies on f and g having the same shape)
            g=g.T
    a,b=f.shape # if eval makes it to this point, f.shape==g.shape is True

    fp=np.zeros((2*a,2*b)) # holders for padded versions (wraparound-safe...)
    gp=np.zeros((2*a,2*b))
    fp[:a,:b]=f # populate the zero-padded versions
    gp[:a,:b]=g

    result_p=convolve(fp,gp)
    result=result_p[:a,:b]
    return result

def calc_par_vec(kpar,sigLoS,r0): # this was initially separated into its own function because there was a lot more going on under the hood when my integral was wrong
    """
    LoS term of the cylindrically binned window function, calculated for the k-parallel modes in the survey
    """
    expterm=twopi*sigLoS**2*np.exp(-kpar**2*sigLoS**2)
    return expterm

def calc_perp_vec(kperp,r0,fwhmbeam,beamtype="Gaussian"):
    """
    flat sky approximation sky plane term of the cylindrically binned window function, calculated for the k-perp modes in the survey
    """
    beamtype=beamtype.lower()
    if beamtype=="gaussian":
        alpha=ln2/(fwhmbeam*r0)**2
        vec=np.exp(-kperp**2/(2*alpha))
    else: 
        assert(1==0), "only a Gaussian beam is currently supported"
    return vec

def W_cyl_binned(kpar,kperp,sigLoS,r0,fwhmbeam,save=False,savename="test",beamtype="Gaussian"):
    """
    wrapper to multiply the LoS and flat sky approximation sky plane terms of the cylindrically binned window function, for the grid described by the k-parallel and k-perp modes of the survey of interest
    """
    par_vec=calc_par_vec(kpar,sigLoS,r0)
    perp_vec=calc_perp_vec(kperp,r0,fwhmbeam,beamtype=beamtype)
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

def calc_Wcont(kpar,kperp,sigLoS,r0,fwhmbeam,epsLoS,epsbeam,save="False",savename=None,beamtype="Gaussian"): 
    """
    calculate the "contaminant" windowing amplitude that will help give rise to the so-called "contaminant power"
    """
    Wtrue=   W_cyl_binned(kpar,kperp,sigLoS,           r0,fwhmbeam,            save=save,savename=savename,beamtype=beamtype)
    Wthought=W_cyl_binned(kpar,kperp,sigLoS*(1-epsLoS),r0,fwhmbeam*(1-epsbeam),save=save,savename=savename,beamtype=beamtype) # FOR NOW: BAKED IN THAT THE "THOUGHT" WIDTH RESPONSE PARAMS ARE UNDERESTIMATES FOR POSITIVE EPS
    return Wtrue-Wthought

def calc_Pcont_cyl(kpar,kperp,sigLoS,r0,fwhmbeam,pars,epsLoS,epsbeam,z,n_sph_modes,beamtype="Gaussian",save=False,savename=None): 
    """
    calculate the cylindrically binned "contaminant power," following from the true and perceived window functions
    """
    Wcont=calc_Wcont(kpar,kperp,sigLoS,r0,fwhmbeam,epsLoS,epsbeam,save=save,savename=savename,beamtype=beamtype)
    _,_,P=unbin_to_Pcyl(kpar,kperp,z,pars=pars,n_sph_modes=n_sph_modes)
    Pcont=higher_dim_conv(Wcont,P)
    return Pcont

def calc_Pcont_cyl_asym_resp(pars,z,kpar,kperp,sigLoS,epsLoS,r0,beamfwhm_x,beamfwhm_y,eps_x,eps_y,n_realiz,ncubevox=100,n_sph_modes=500,recalc_Pcont=False):
    """
    calculate a cylindrically binned Pcont from an average over the power spectra formed from cylindrically-asymmetric-response-modulated brightness temp fields for a cosmological case of interest
    (you can still form a cylindrical summary statistic from brightness temp fields encoding effects beyond this symmetry)

    args
    beamfwhm_x = FWHM of the beam in one       polarization direction
    beamfwhm_y = "                 " the other "                    "
    eps_x      = fractional uncertainty in beamfwhm_x
    eps_y      = fractional uncertainty in beamfwhm_y
    n_realiz   = number of times to iterate the "generate a random realization of the Tb cube" -> "multiply by the beam in config space" -> "form PS with cylindrical bins of survey" step
    ncubevox   = number of voxels per side to use when constructing random realization Tb cubes

    returns
    contaminant power, calculated as an average over the "beam-modulated" PS resulting from the loop iterated n_realiz times 
    """
    h=pars[0]/100 # this relies on H0 being the first parameter in pars ... and more fundamentally, H0 being in the forecast at all
    kmin=np.sqrt(kpar[0]**2+ kperp[0]**2)
    kmax=np.sqrt(kpar[-1]**2+kperp[-1]**2)
    ksph,Ptrue=get_mps(pars,z,minkh=kmin/h,maxkh=kmax/h,n_sph_modes=500)

    nkpar=len(kpar)
    nkperp=len(kperp)
    Pconts=np.zeros((nkpar,nkperp,n_realiz)) # holder for the cylindrically binned power spectra (will make it easy to average later)
    Lcube=int(2*sigLoS) # should be some multiple of 2*sigLoS?? 3x for 3sigma level?? or just 1x because that's how I motivated constructing it from Dchi and Dclo??
    sigLoS_instances=     np.random.normal(loc=sigLoS,     scale=epsLoS*sigLoS,    size=n_realiz) # np.random.normal(loc=0.0, scale=1.0, size=None),, loc~mu, scale~sigma
    beamfwhm_x_instances= np.random.normal(loc=beamfwhm_x, scale=eps_x*beamfwhm_x, size=n_realiz)
    beamfwhm_y_instances= np.random.normal(loc=beamfwhm_y, scale=eps_y*beamfwhm_y, size=n_realiz)
    for i in range(n_realiz):
        rcube,Tcube,rmags = ips(Ptrue,ksph,Lcube,ncubevox) # rgrid,T=ips(P,k,Lsurvey,nfvox)
        if (i==0): # initialize the instrument response
            X,Y,Z=np.meshgrid(rmags,rmags,rmags,indexing="ij") # I didn't specify ij indexing in the meshgridding internal to ips(), but I don't think it matters at the moment since everything in the cube is so statistically isotropic,, (could eventually circle back there to be really rigorous)
        instrument_response=np.exp(-Z**2/(2*sigLoS_instances[i]**2)-ln2*((X/beamfwhm_x_instances[i])**2+(Y/beamfwhm_y_instances[i])**2)/r0**2) # mathematically equivalent to offsetting Z down the line of sight by r0 and then using the original functional form with the subtraction, but with fewer steps
        response_aware_cube=Tcube*instrument_response # configuration-space multiplication
        # print("response_aware_cube instance=",response_aware_cube)
        kcont,Pcont=ps_userbin(response_aware_cube,ksph,Lcube) # ps_userbin(T, kbins, Lsurvey) returns P() ... which returns [k,P]
        # print("Pcont instance=",Pcont)
        kpargrid,kperpgrid,Pcyl=unbin_to_Pcyl_custom(kpar,kperp,kcont,Pcont) # kpargrid,kperpgrid,Pcyl = unbin_to_Pcyl_custom(kpar,kperp,k,Psph)
        Pconts[:,:,i]=Pcyl
    Pcont_avg=np.mean(Pconts,axis=2)
    np.save("Pcont_avg.npy",Pcont_avg)
    return Pcont_avg

def unbin_to_Pcyl(kpar,kperp,z,pars=pars_Planck18,n_sph_modes=500):  
    """
    interpolate a spherically binned CAMB MPS to provide MPS values for a cylindrically binned k-grid of interest (nkpar x nkperp)
    """
    h=pars[0]/100.
    kmin=np.sqrt(kpar[0]**2+kperp[0]**2)
    kmax=np.sqrt(kpar[-1]**2+kperp[-1]**2)
    k,Psph=get_mps(pars,z,minkh=kmin/h,maxkh=kmax/h,n_sph_modes=n_sph_modes)
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


def unbin_to_Pcyl_custom(kpar,kperp,k,Psph): 
    """
    same logic as the version not called _custom, but BYO power spectrum

    args
    k    = scalar (i.e. spherically binned) k-modes at which your 1D power spectrum is sampled
    Psph = 1D power spectrum to de-bin to cylindrical

    returns
    interpolation-fuelled de-binning of a user-provided spherically binned power spectrum to a set of cylindrical bins of interest
    """
    if (Psph.shape[0]==1): # if transposed compared to what I want,
        Psph=Psph.reshape((Psph.shape[1],)) # transpose it
    n_sph_modes=len(Psph) # now, backing out the length is unambiguous
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
def get_mps(pars,z,minkh=1e-4,maxkh=1,n_sph_modes=500):
    """
    get matter power spectrum from CAMB

    args
    minkh  = min value of k/h at which to calculate the MPS
    maxkh  = max value of k/h at which to calculate the MPS 
    """
    z=[z]
    H0=pars[0]
    h=H0/100.
    ombh2=pars[1]
    omch2=pars[2]
    As=pars[3]*scale
    ns=pars[4]

    pars=camb.set_params(H0=H0, ombh2=ombh2, omch2=omch2, ns=ns, mnu=0.06,omk=0)
    pars.InitPower.set_params(As=As,ns=ns,r=0)
    pars.set_matter_power(redshifts=z, kmax=maxkh*h)
    lin=True
    results = camb.get_results(pars)
    pars.NonLinear = model.NonLinear_none
    kh,z,pk=results.get_matter_power_spectrum(minkh=minkh,maxkh=maxkh,npoints=n_sph_modes)
    return kh,pk

def cyl_partial(pars,z,n,dpar,kpar,kperp,n_sph_modes=200,ftol=1e-6,eps=1e-16,maxiter=5):  
    """
    args
    n           = take the partial derivative WRT the nth parameter in p
    ftol        = fractional tolerance relating to the scale of the function (defined for points of interest)
    eps         = tiny offset factor to protect against numerical division-by-zero errors
    maxiter     = maximum number of times to let the step size optimization attempt recurse before "giving up" and using the most recent guess

    returns
    cylindrically binned matter power spectrum partial WRT one cosmo parameter (nkpar x nkperp)
    """
    done=False
    iter=0
    nkpar=len(kpar)
    nkperp=len(kperp)
    dparn=dpar[n]
    pcopy=pars.copy()
    pndispersed=pcopy[n]+np.linspace(-2,2,5)*dparn

    pcopy=pars.copy()
    _,_,Pcyl_base=unbin_to_Pcyl(kpar,kperp,z,pars=pcopy,n_sph_modes=n_sph_modes)
    P0=np.mean(np.abs(Pcyl_base))+eps
    tol=ftol*P0 # generalizes tol=ftol*f0 from 512

    pcopy[n]=pcopy[n]+2*dparn
    _,_,Pcyl_2plus=unbin_to_Pcyl(kpar,kperp,z,pars=pcopy,n_sph_modes=n_sph_modes)
    pcopy=pars.copy()
    pcopy[n]=pcopy[n]-2*dparn
    _,_,Pcyl_2minu=unbin_to_Pcyl(kpar,kperp,z,pars=pcopy,n_sph_modes=n_sph_modes)
    deriv1=(Pcyl_2plus-Pcyl_2minu)/(4*dpar[n])

    pcopy[n]=pcopy[n]+dparn
    _,_,Pcyl_plus=unbin_to_Pcyl(kpar,kperp,z,pars=pcopy,n_sph_modes=n_sph_modes)
    pcopy=pars.copy()
    pcopy[n]=pcopy[n]-dparn
    _,_,Pcyl_minu=unbin_to_Pcyl(kpar,kperp,z,pars=pcopy,n_sph_modes=n_sph_modes)
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

def build_cyl_partials(pars,z,n_sph_modes,kpar,kperp,dpar):
    """
    builds a ncp x nkpar x nkperp array of the partials of the cylindrically binned MPS WRT each cosmo param in the forecast
    """
    nkpar=len(kpar)
    nkperp=len(kperp)
    nprm=len(pars)
    V=np.zeros((nprm,nkpar,nkperp))
    for n in range(nprm):
        V[n,:,:]=cyl_partial(pars,z,n,dpar,kpar,kperp,n_sph_modes=n_sph_modes)
    return V

def bias(partials,unc, kpar,kperp,sigLoS,r0,fwhmbeam0,pars,epsLoS,epsbeam0,z,n_sph_modes,beamtype="Gaussian",save=False,savename=None,cyl_sym_resp=True, fwhmbeam1=1e-3,epsbeam1=0.1,n_realiz=10,ncubevox=100,recalc_Pcont=False):
    """
    args
    partials = ncp x nkpar x nkperp array where each slice of constant 0th (nprm) index is an nkpar x nkperp array of the MPS's partial WRT a particular parameter in the forecast
    unc      = nkpar x nkperp array describing the standard deviations at each cylindrically binned k-mode

    returns
    (ncp,) vector of biases resulting from beam mismodelling for the parameters of interest in the forecast
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
    if cyl_sym_resp:
        Pcont=calc_Pcont_cyl(kpar,kperp,sigLoS,r0,fwhmbeam0,pars,epsLoS,epsbeam0,z,n_sph_modes,save=save,savename=savename,beamtype=beamtype)
    else: 
        Pcont=calc_Pcont_cyl_asym_resp(pars,z,
                                       kpar,kperp,
                                       sigLoS,epsLoS,r0,fwhmbeam0,fwhmbeam1,epsbeam0,epsbeam1,
                                       n_realiz,ncubevox=ncubevox,n_sph_modes=n_sph_modes,
                                       recalc_Pcont=recalc_Pcont) # calc_Pcont_cyl_asym_resp(pars,z,kpar,kperp,sigLoS,epsLoS,r0,beamfwhm_x,beamfwhm_y,eps_x,eps_y,n_realiz,ncubevox=100,n_sph_modes=500,recalc_Pcont=False)
    plt.figure()
    plt.imshow(Pcont, origin="lower",extent=[kpar[0],kpar[-1],kperp[0],kperp[-1]]) # origin="lower",extent=[L_lo,R_hi,T_lo,B_hi]
    plt.xlabel("k$_{||}$ (Mpc$^{-1}$)")
    plt.ylabel("k$_{\perp}$ (Mpc$^{-1}$)")
    plt.title("Pcont")
    plt.colorbar()
    plt.savefig("Pcont.png")
    plt.show()
    Pcont_div_sigma=Pcont/unc
    B=np.einsum("jk,ijk->i",Pcont_div_sigma,V)
    bias=(np.linalg.inv(F)@B).reshape((F.shape[0],))
    return bias

def printparswbiases(pars,parnames,biases):
    """
    args
    parnames = (ncp,) vector of strings: names of the parameters in the forecast (assumed to be in the same order as pars)
    biases   = (ncp,) vector: biases in estimating the cosmo params in the forecast resulting from inadvertent beam mismodelling (in the same order as pars and parnames)

    returns
    None (fcn prints a formatted summary of the forecasting pipeline calculation)
    """
    print("\n\nbias calculation results for the survey described above.................................")
    print("........................................................................................")
    for p,par in enumerate(pars):
        print('{:12} = {:-10.3e} with bias {:-12.5e} (fraction = {:-10.3e})'.format(parnames[p], par, biases[p], biases[p]/par))
    return None