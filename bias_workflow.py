import numpy as np
from matplotlib import pyplot as plt
import pygtc

def Pgau(k,pars):
    mup,sigp,Ap=pars
    return Ap*np.exp(-(k-mup)**2/(2*sigp**2))

def build_partials(m,p,dpar,nmodes):
    '''
    m      = vector of modes you want to sample your power spectrum at (nmodes x 1)
    p      = vector of cosmological parameters (npar x 1)
    dpar   = vector (since you might want dif step sizes for dif params) of step sizes (npar x 1)
    nmodes = [scalar] number of modes in the spectrum - could be l-modes for CMB, k-modes for 21 cm, etc.
    '''
    nprm=len(p)
    V=np.zeros((nmodes,nprm))
    for n in range(nprm):
        V[:,n]=Ppartial(m,p,n,dpar)
    return V

def fisher(partials,unc):
    '''
    partials = nmodes x nprm array where each column is an nmodes x 1 vector of the PS's partial WRT a dif param
    unc      = nmodes x 1 vector of standard deviations at each mode (could be k-mode, l-mode, etc.)
    '''
    V=0.0*partials # want the same shape
    nprm=partials.shape[1]
    for i in range(nprm):
        V[:,i]=partials[:,i]/unc
    return V.T@V

def Ppartial(m,p,n,dpar):
    '''
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

def printparswbiases(pars,biases):
    for p,par in enumerate(pars):
        print(parnames[p].ljust(10),'=',str(par).ljust(5),'with bias',str(round(biases[p],2)).ljust(5))
    return None

nk=132
kvec=np.linspace(0.05,0.7,nk)
mup=0.11
sigp=0.2 # width of the Gaussian I'm using as a power spectrum
ampp=431
pars=[mup,sigp,ampp]
parnames=['mean','std dev','amp']
npars=len(pars)
Ptrue=Pgau(kvec,pars)
sigk=0.05*np.cos(2*np.pi*(kvec-kvec[0])/(kvec[-1]-kvec[0]))+0.01 # uncertainty in the power spectrum at each k-mode

def Wmat(kvec,sigma): # same as in the window_meshgrid.py script
   k,kp=np.meshgrid(kvec,kvec)
   return (2*np.pi*sigma**2)**3*np.exp(-sigma**2*(k-kp)**2/2.)

sigw=1e-9 # window width (following the form I derived)
Pcont=Wmat(kvec,sigw)@Ptrue-Ptrue
dpar=1e-9*np.ones(npars)
Ppartials=build_partials(kvec,pars,dpar,nk) # build_partials(m,p,dpar,nmodes)
F=fisher(Ppartials,sigk)

B=(Pcont/sigk)@Ppartials # should be a three-element vector (verified by printing the shape) !!
Finv=np.linalg.inv(F)
b=Finv@B # should be a three-element vector (verified by printing the shape) !!

printparswbiases(pars,b)