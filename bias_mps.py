import numpy as np
from matplotlib import pyplot as plt
import pygtc
from scipy.special import erf
import camb
from camb import model

def Pgau(k,pars): # STANDS IN FOR SOMETHING MORE PHYSICALLY INTERESTING AND REALISTIC LIKE A CAMB MPS ... STICK WITH THIS FOR NOW BECAUSE MY TESTS WILL RUN MORE QUICKLY
    mup,sigp,Ap=pars
    return Ap*np.exp(-(k-mup)**2/(2*sigp**2))

def build_partials(m,getP,p,dpar,nmodes):
    '''
    m      = vector of modes you want to sample your power spectrum at (nmodes x 1)
    p      = vector of cosmological parameters (npar x 1)
    dpar   = vector (since you might want dif step sizes for dif params) of step sizes (npar x 1)
    nmodes = [scalar] number of modes in the spectrum - could be l-modes for CMB, k-modes for 21 cm, etc.
    '''
    # print('in build_partials, p=',p)
    nprm=len(p)
    V=np.zeros((nmodes,nprm))
    for n in range(nprm):
        V[:,n]=getP(m,p,n,dpar) # THIS CALL IS WRONG?? ... for CAMB, I call build_partials with getP=CAMBpartial, which is called as CAMBpartial(p,zs,n,dpar)
    return V

def buildCAMBpartials(p,z,NMODES,dpar):
    '''
    m      = vector of modes you want to sample your power spectrum at (nmodes x 1)
    p      = vector of cosmological parameters (npar x 1)
    dpar   = vector (since you might want dif step sizes for dif params) of step sizes (npar x 1)
    nmode = [scalar] number of modes in the spectrum - could be l-modes for CMB, k-modes for 21 cm, etc.
    '''
    # print('in buildCAMBpartials, p=',p)
    nprm=len(p)
    V=np.zeros((NMODES,nprm))
    for n in range(nprm):
        V[:,n]=CAMBpartial(p,z,n,dpar,nmodes=NMODES) # THIS CALL IS WRONG?? ... for CAMB, I call build_partials with getP=CAMBpartial, which is called as CAMBpartial(p,zs,n,dpar)
    print('at the end of buildCAMBpartials, V.shape=',V.shape)
    return V

def fisher(partials,unc):
    '''
    partials = nmodes x nprm array where each column is an nmodes x 1 vector of the PS's partial WRT a dif param
    unc      = nmodes x 1 vector of standard deviations at each mode (could be k-mode, l-mode, etc.)
    '''
    # print('at the beginning of fisher, partials.shape=',partials.shape,'and unc.shape=',unc.shape)
    V=0.0*partials # want the same shape
    nprm=partials.shape[1]
    for i in range(nprm):
        V[:,i]=partials[:,i]/unc
    return V.T@V

def Ppartial(m,p,n,dpar):
    ''' GOOD FOR A POWER SPECTRUM WITH A SIMPLE[R THAN CAMB] ANALYTICAL FORM
    m    = vector of modes (k-, l-, etc.) you want to sample your ps at
    p    = vector of cosmological parameters (npar x 1)
    n    = take the partial derivative WRT the nth parameter in p
    dpar = vector (you might want dif step sizes for dif params) of step sizes (npar x 1)
    '''
    pk=Pgau(m,p)
    npts=len(m)
    pcopy=p.copy()
    pcopy[n]=pcopy[n]+dpar[n]
    pkp=Pgau(m,pcopy)
    fplus=np.array(pkp[:npts])
    pcopy=p.copy()
    pcopy[n]=pcopy[n]-dpar[n]
    pkm=Pgau(m,pcopy)
    fminu=np.array(pkm[:npts])
    return ((fplus-fminu)/(2*dpar[n])).reshape((npts,))

def cornerplot(fishermat,params,pnames,nsamp=10000,savename=None):
    cov=np.linalg.inv(fishermat)
    samples=np.random.multivariate_normal(params,cov,size=nsamp)
    GTC=pygtc.plotGTC(chains=samples,
                      paramNames=pnames,
                      truths=tuple(params),
                      plotName=savename)
    return None

def printparswbiases(pars,parnames,biases):
    print('in printparswbiases, pars.shape=',pars.shape,', pars=',pars,'\nbiases.shape=',biases.shape,', biases=',biases)
    print()
    for p,par in enumerate(pars):
        # print('p=',p,'; par=',par)
        # print(parnames[p],'=',str(par),'with bias',str(round(biases[p],2)))
        # print(parnames[p].ljust(12),'=',str(par).ljust(5),'with bias',str(round(biases[p],2)).ljust(5))
        print('{:12} = {:-10.3e} with bias {:-10.5e}'.format(parnames[p], par, biases[p]))
    return None

def Wmat(kvec,sigma): 
   k,kp=np.meshgrid(kvec,kvec)
   return 2*np.pi*sigma**2*np.exp(-sigma**2*(k-kp)**2)

def Wmatskew(kvec,sigma): 
   k,kp=np.meshgrid(kvec,kvec)
   return 2*np.pi*sigma**2*np.exp(-sigma**2*(k-kp)**2)*(1+erf(5*k))

scale=1e-9
def get_mps(pars,zs,minkh=1e-4,maxkh=1,npts=200):
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
    # print('in get_mps, pars=',pars)
    H0=pars[0]
    ombh2=pars[1]
    omch2=pars[2]
    As=pars[3]*scale ######## find a way to bake into the MPS calc?
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

def CAMBpartial(p,zs,n,dpar,nmodes=200):
    '''
    p    = vector of cosmological parameters (npar x 1)
    zs   = tuple of redshifts where we're interested in calculating the MPS
    n    = take the partial derivative WRT the nth parameter in p
    dpar = vector (you might want dif step sizes for dif params) of step sizes (npar x 1)
    '''
    # print('in CAMBpartial, pars=',p)
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

def bias(F,B):
    return np.linalg.inv(F)@B

nk=132
kvec=np.linspace(0.05,0.7,nk)
mup=0.11
sigp=0.2 # width of the Gaussian I'm using as a power spectrum
ampp=431
pars=[mup,sigp,ampp]
parnames=['mean','std dev','amp']
npars=len(pars)
dpar=1e-9*np.ones(npars)
Ptrue=Pgau(kvec,pars)
sigk=0.05*np.cos(2*np.pi*(kvec-kvec[0])/(kvec[-1]-kvec[0]))+0.01 # uncertainty in the power spectrum at each k-mode ... just a toy case sinusoidally more sensitive in the middle but offset vertically so there is a positive uncertainty at each k

CAMBpars=np.asarray([67.7,0.022,0.119,2.1e-9, 0.97])
CAMBparnames=['H_0','Omega_b h^2','Omega_c h^2','A_S','n_s']
CAMBparnames_LaTeX=['$H_0$','$\Omega_b h^2$','$\Omega_c h^2$','$A_S$','$n_s$']
CAMBpars[3]/=scale
nprm=len(CAMBpars) # number of parameters
CAMBdpar=1e-3*np.ones(nprm)
CAMBdpar[3]*=scale
ztest=7.4
# print('in main, CAMBpars=', CAMBpars)
CAMBk,CAMBPtrue=get_mps(CAMBpars,ztest,npts=nk)
# print('after calling get_mps from main, CAMBpars=',CAMBpars)
CAMBnpars=len(CAMBpars)
calcCAMBPpartials=True

sigw=1e-1 # window width (following the form I derived)
W=Wmat(kvec,sigw)
epssigws=np.logspace(-5,0,10) # multiplicative prefactor: "what fractional error do you have in your knowledge of the beam width"

for epssigw in epssigws:
    print('\nepssigw=',epssigw)
    Wthought=Wmat(kvec,(1+epssigw)*sigw)

    # TOY GAUSSIAN POWER SPECTRUM CASE
    # Pcont=(W-Wthought)@Ptrue
    # Ppartials=build_partials(kvec,Ppartial,pars,dpar,nk) # build_partials(m,getP,p,dpar,nmodes)
    # F=fisher(Ppartials,sigk)
    # B=(Pcont/sigk)@Ppartials # should be a three-element vector (verified by printing the shape) !!
    # b=bias(F,B)
    # printparswbiases(pars,parnames,b)

    # CAMB MATTER POWER SPECTRUM CASE
    CAMBPcont=(W-Wthought)@Ptrue
    # print('in the epssigws loop, before establishing CAMBPpartials, CAMBpars=',CAMBpars)
    if calcCAMBPpartials:
        # print('about to calculate CAMBPpartials')
        CAMBPpartials=buildCAMBpartials(CAMBpars,ztest,nk,CAMBdpar) # buildCAMBpartials(p,z,nmodes,dpar)
        np.save('cambppartials.npy',CAMBPpartials)
        # print('calculated and saved CAMBPpartials')
    else:
        CAMBPpartials=np.load('cambppartials.npy')
    print('established CAMBPpartials without issue')
    CAMBF=fisher(CAMBPpartials,sigk)
    CAMBB=(CAMBPcont/sigk)@CAMBPpartials
    CAMBb=bias(CAMBF,CAMBB)

    CAMBpars2=CAMBpars.copy()
    CAMBpars2[3]*=scale
    CAMBb2=CAMBb.copy()
    CAMBb2[3]*=scale
    printparswbiases(CAMBpars2,CAMBparnames,CAMBb2)

