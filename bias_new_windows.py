import numpy as np
from matplotlib import pyplot as plt
import pygtc
from scipy.special import j1 # first-order Bessel function of the first kind
from scipy.integrate import quad,dblquad
import camb
from camb import model

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

# quad(   func, a, b,             args=(), full_output=0, epsabs=1.49e-08, epsrel=1.49e-08, limit=50, points=None, weight=None, wvar=None, wopts=None, maxp1=50, limlst=50, complex_func=False) # ** "If func takes many arguments, it is integrated along the axis corresponding to the first argument."
# dblquad(func, a, b, gfun, hfun, args=(), epsabs=1.49e-08, epsrel=1.49e-08)
# nquad(  func, ranges, args=None, opts=None, full_output=False)

# argstuple, optional
# Extra arguments to pass to func.

# if you have a function fcn(var,par0,par1,...,parnm1), you need to specify the values of par0 through parnm1 using args as:
# args=(par0val,par1val,...,parnm1val,)

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
    integral,_=quad(theta_arg_real,0,np.pi,args=(thetak,thetakp,))
    return integral
def inner_theta_integral_imag(thetak,thetakp):
    integral,_=quad(theta_arg_imag,0,np.pi,args=(thetak,thetakp,))
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
    integral,_=quad(phi_arg_real,0,2*np.pi,args=(phik,phikp,))
    return integral
def inner_phi_integral_imag(phik,phikp):
    integral,_=quad(phi_arg_imag,0,2*np.pi,args=(phik,phikp,))
    return integral
def phi_k_and_kp_arg(phik,phikp):
    return inner_phi_integral_real(phik,phikp)**2+inner_phi_integral_imag(phik,phikp)**2

def W_binned_airy_beam(rk,rkp,sig):
    r_like=inner_r_integral_real(rk,rkp,sig)**2+inner_r_integral_imag(rk,rkp,sig)**2
    theta_like=dblquad(theta_k_and_kp_arg,0,np.pi,0,np.pi) # thetak and thetakp aren't args; they're variables of integration
    phi_like=dblquad(phi_k_and_kp_arg,0,2*np.pi,0,2*np.pi) # phik and phikp are vars of integration
    print('in W_binned_airy_beam: done calculating r_like')
    return 4*pi**2*r_like*theta_like*phi_like 

# ######### OLD VERSION WITH NOT ENOUGH REAL/IMAG SEPARATION FROM TUES MARCH 11TH
# def theta_arg(theta,thetak,thetakp):
#     return np.exp(-1j*(thetak-thetakp)*theta)*(j1(theta)/theta)**2*np.sin(theta)
# def theta_k_and_kp_arg(thetak,thetakp):
#     inner_theta_integral,_=quad(theta_arg,0,pi,args=(thetak,thetakp,))
#     return np.abs(inner_theta_integral)**2*np.sin(thetak)*np.sin(thetakp)
# theta_like=dblquad(theta_k_and_kp_arg,0,pi,0,pi)
# print('in main: done calculating theta_like')

# def phi_arg(phi,phik,phikp):
#     return np.exp(-1j*(phik-phikp)*phi)
# def phi_k_and_kp_arg(phik,phikp):
#     inner_phi_integral,_=quad(phi_arg,0,twopi,args=(phik,phikp,))
#     return np.abs(inner_phi_integral)**2
# phi_like=dblquad(phi_k_and_kp_arg,0,twopi,0,twopi)
# print('in main: done calculating phi_like')

# def W_binned_airy_beam(rk,rkp,sig):
#     inner_r_integral,_=quad(r_arg,0,np.infty,args=(rk,rkp,sig,))
#     r_like=inner_r_integral**2
#     print('in W_binned_airy_beam: done calculating r_like')
#     return 4*pi**2*r_like*theta_like*phi_like # boo global variables or whatever. but alas. I am a physicist at heart. 
# #########

npts=799
rkmax=100.
rk_test=np.linspace(0,rkmax,npts)
rkp_test=np.copy(rk_test)
sig_test=rkmax/2
W_binned_airy_beam_test=W_binned_airy_beam(rk_test,rkp_test,sig_test)

plt.figure()
plt.imshow(W_binned_airy_beam_test)
plt.colorbar()
plt.xlabel('rk [dimensions of 1/L; units not specified]')
plt.ylabel('rkp [dimensions of 1/L; units not specified]')
plt.title('Visualization check: spherical harmonic binning of an Airy beam')
plt.show()

# # reality check ... does this seem like the Bessel function I meant to use and the Airy beam I meant to describe?
# lim=25
# npts=1000
# x=np.linspace(-lim,lim,npts)
# plt.figure()
# plt.plot(x,(j1(x)/x)**2)
# plt.show()