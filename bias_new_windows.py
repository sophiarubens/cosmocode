import numpy as np
from matplotlib import pyplot as plt
import pygtc
from scipy.special import j1 # first-order Bessel function of the first kind
from scipy.integrate import quad,dblquad
import camb
from camb import model
import time

infty=np.infty
pi=np.pi
twopi=2.*pi

scale=1e-9
def get_mps(pars,zs,minkh=1e-4,maxkh=1,npts=200): # < CAMBpartial < buildCAMBpartials
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
    ombh2=pars[1]
    omch2=pars[2]
    As=pars[3]*scale
    ns=pars[4]

    pars=camb.set_params(H0=H0, ombh2=ombh2, omch2=omch2, ns=ns, mnu=0.06,omk=0)
    pars.InitPower.set_params(As=As,ns=ns,r=0)
    pars.set_matter_power(redshifts=zs, kmax=2.0)
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
    p    = vector of cosmological parameters (npar x 1)
    zs   = tuple of redshifts where we're interested in calculating the MPS
    n    = take the partial derivative WRT the nth parameter in p
    dpar = vector (you might want dif step sizes for dif params) of step sizes (npar x 1)
    '''
    kh,pk=get_mps(p,zs,npts=nmodes) # model should be get_spec for the unperturbed params
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

def fisher(partials,unc): # output to cornerplot or bias
    '''
    partials = nmodes x nprm array where each column is an nmodes x 1 vector of the PS's partial WRT a dif param
    unc      = nmodes x 1 vector of standard deviations at each mode (could be k-mode, l-mode, etc.)
    '''
    V=0.0*partials # want the same shape
    nprm=partials.shape[1]
    for i in range(nprm):
        V[:,i]=partials[:,i]/unc
    return V.T@V

def cornerplot(fishermat,params,pnames,nsamp=10000,savename=None):
    cov=np.linalg.inv(fishermat)
    samples=np.random.multivariate_normal(params,cov,size=nsamp)
    GTC=pygtc.plotGTC(chains=samples,
                      paramNames=pnames,
                      truths=tuple(params),
                      plotName=savename)
    return None

def Wmat(kvec,sigma): # USE THIS AS A TEMPLATE FOR THE MORE-COMPLICATED NUMERICAL VERSION
   k,kp=np.meshgrid(kvec,kvec)
   return twopi*sigma**2*np.exp(-sigma**2*(k-kp)**2)

def bias(F,B):
    return np.linalg.inv(F)@B

def printparswbiases(pars,parnames,biases):
    for p,par in enumerate(pars):
        print('{:12} = {:-10.3e} with bias {:-12.5e}'.format(parnames[p], par, biases[p]))
    return None

# NUMERICALLY INTEGRATING TO OBTAIN A SPHERICALLY HARMONICALLY BINNED WINDOW FUNCTION
# THIS USES AN AIRY BEAM WITH GAUSSIAN CHROMATICITY ... WORK THROUGH THE BUGS WITH THIS, BUT GENERALIZE LATER
# ####################################################### START OF COMPLEX_FUNC=TRUE VERSION
# def r_arg(r,rk,rkp,sig):
#     return np.exp(-1j*(rk-rkp)*r-((r**2)/(2*sig**2)))*r**2
# def inner_r_integral(rk,rkp,sig):
#     integral,_=quad(r_arg,0,np.infty,args=(rk,rkp,sig,),complex_func=True)
#     return integral

# def theta_arg(theta,thetak,thetakp):
#     return np.exp(-1j*(thetak-thetakp)*theta)*(j1(theta)/theta)**2*np.sin(theta)
# def inner_theta_integral(thetak,thetakp):
#     integral,_=quad(theta_arg,0,pi,args=(thetak,thetakp,),complex_func=True)
#     return integral
# def theta_k_and_kp_arg(thetak,thetakp):
#     return np.abs(inner_theta_integral(thetak,thetakp))**2*np.sin(thetak)*np.sin(thetakp) # apparently it makes all the difference to do np.abs()**2 instead of manually taking ()**2

# def phi_arg(phi,phik,phikp):
#     return np.exp(-1j*(phik-phikp)*phi)
# def inner_phi_integral(phik,phikp):
#     integral,_=quad(phi_arg,0,twopi,args=(phik,phikp,),complex_func=True)
#     return integral
# def phi_k_and_kp_arg(phik,phikp):
#     return np.abs(inner_phi_integral(phik,phikp))**2

# def W_binned_airy_beam_entry(rk,rkp,sig): # ONE ENTRY in the kind of W_binned square array that is useful to build
#     r_like=np.abs(inner_r_integral(rk,rkp,sig))**2
#     print('r_like=',r_like)
#     theta_like,_=dblquad(theta_k_and_kp_arg,0,pi,0,pi) # thetak and thetakp aren't args; they're variables of integration
#     print('theta_like=',theta_like)
#     phi_like,_=dblquad(phi_k_and_kp_arg,0,twopi,0,twopi) # phik and phikp are vars of integration
#     print('phi_like=',phi_like)
#     return 4*pi**2*r_like*theta_like*phi_like 
# ####################################################### END OF COMPLEX_FUNC=TRUE VERSION

# NUMERICALLY INTEGRATING TO OBTAIN A SPHERICALLY HARMONICALLY BINNED WINDOW FUNCTION
# THIS USES AN AIRY BEAM WITH GAUSSIAN CHROMATICITY ... WORK THROUGH THE BUGS WITH THIS, BUT GENERALIZE LATER
def r_arg_real(r,rk,rkp,sig):
    arg=np.exp(-1j*(rk-rkp)*r-((r**2)/(2*sig**2)))*r**2
    return arg.real
def r_arg_imag(r,rk,rkp,sig):
    arg=np.exp(-1j*(rk-rkp)*r-((r**2)/(2*sig**2)))*r**2
    return arg.imag
def inner_r_integral_real(rk,rkp,sig):
    integral,_=quad(r_arg_real,0,np.infty,args=(rk,rkp,sig,))
    return integral
def inner_r_integral_imag(rk,rkp,sig):
    integral,_=quad(r_arg_imag,0,np.infty,args=(rk,rkp,sig,))
    return integral

def theta_arg_real(theta,thetak,thetakp):
    arg=np.exp(-1j*(thetak-thetakp)*theta)*(j1(theta)/theta)**2*np.sin(theta)
    return arg.real
def theta_arg_imag(theta,thetak,thetakp):
    arg=np.exp(-1j*(thetak-thetakp)*theta)*(j1(theta)/theta)**2*np.sin(theta)
    return arg.imag
def inner_theta_integral_real(thetak,thetakp):
    integral,_=quad(theta_arg_real,0,pi,args=(thetak,thetakp,))
    return integral
def inner_theta_integral_imag(thetak,thetakp):
    integral,_=quad(theta_arg_imag,0,pi,args=(thetak,thetakp,))
    return integral
def theta_k_and_kp_arg(thetak,thetakp):
    inner_theta_integral=inner_theta_integral_real(thetak,thetakp)**2+inner_theta_integral_imag(thetak,thetakp)**2
    return inner_theta_integral*np.sin(thetak)*np.sin(thetakp)

def phi_arg_real(phi,phik,phikp):
    arg=np.exp(-1j*(phik-phikp)*phi)
    return arg.real
def phi_arg_imag(phi,phik,phikp):
    arg=np.exp(-1j*(phik-phikp)*phi)
    return arg.imag
def inner_phi_integral_real(phik,phikp):
    integral,_=quad(phi_arg_real,0,twopi,args=(phik,phikp,))
    return integral
def inner_phi_integral_imag(phik,phikp):
    integral,_=quad(phi_arg_imag,0,twopi,args=(phik,phikp,))
    return integral
def phi_k_and_kp_arg(phik,phikp):
    return inner_phi_integral_real(phik,phikp)**2+inner_phi_integral_imag(phik,phikp)**2

def W_binned_airy_beam_entry(rk,rkp,sig): # ONE ENTRY in the kind of W_binned square array that is useful to build
    r_like=inner_r_integral_real(rk,rkp,sig)**2+inner_r_integral_imag(rk,rkp,sig)**2
    # print('r_like=',r_like)
    theta_like,_=dblquad(theta_k_and_kp_arg,0,pi,0,pi) # thetak and thetakp aren't args; they're variables of integration
    # print('theta_like=',theta_like)
    phi_like,_=dblquad(phi_k_and_kp_arg,0,twopi,0,twopi) # phik and phikp are vars of integration
    # print('phi_like=',phi_like)
    return 4*pi**2*r_like*theta_like*phi_like 

def W_binned_airy_beam(rk_vector,sig,save=True,timeout=200,verbose=False): # accumulate the kind of term we're interested in into a square grid
    earlyexit=False # so far
    t0=time.time()
    npts=len(rk_vector)
    arr=np.zeros((npts,npts))
    element_times=np.zeros(npts**2)
    for i in range(npts):
        for j in range(i,npts):
            t1=time.time()
            arr[i,j]=W_binned_airy_beam_entry(rk_vector[i],rk_vector[j],sig)
            arr[j,i]=arr[i,j] # probably a negligible difference to leave it this way vs. adding an if stmt to manually catch the off-diagonal terms
            t2=time.time()
            element_times[i*npts+j]=t2-t1
            if verbose:
                print('populated entries [',i,',',j,'] and [',j,',',i,'] of W_binned_airy_beam in',t2-t1,'s')
            if(t2-t0>timeout):
                earlyexit=True
                break
    if (save):
        np.save('W_binned_airy_beam_array_'+str(time.time())+'.txt',arr)
    t3=time.time()
    if verbose:
        if earlyexit:
            print('due to time constraints, off-diagonal entries with row AND column indices above',i-1,': were not populated')
        print('evaluation took',t3-t0,'s')
        nonzero_element_times=element_times[np.nonzero(element_times)]
        print('eval time per element:',np.mean(nonzero_element_times),'+/-',np.std(nonzero_element_times)) # x2 since half the array doesn't get populated ... easier than using nan-aware quantities
    return arr

npts=15
rkmax=100.
rk_test=np.linspace(0,rkmax,npts)
sig_test=rkmax/2
W_binned_airy_beam_array_test=W_binned_airy_beam(rk_test,sig_test,verbose=True)

plt.figure()
plt.imshow(W_binned_airy_beam_array_test,extent=[rk_test[0],rk_test[-1],rk_test[-1],rk_test[0]]) # L,R,B,T
plt.colorbar()
plt.xlabel('scalar k; arbitrary units w/ dimensions of 1/L')
plt.ylabel('scalar k-prime; arbitrary units w/ dimensions of 1/L')
plt.title('Visualization check: spherical harmonic binning of an Airy beam')
plt.tight_layout()
plt.show()