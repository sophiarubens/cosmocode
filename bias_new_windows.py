import sys
sys.path.append('/Users/sophiarubens/Downloads/research/code/param_bias/')
import numpy as np
from matplotlib import pyplot as plt
import pygtc
from scipy.special import j1 # first-order Bessel function of the first kind
from scipy.integrate import quad,dblquad
import camb
from camb import model
import time
from calculate_airy_gaussian_window import *
from cosmo_distances import *

infty=np.infty
pi=np.pi
twopi=2.*pi
nu_rest_21=1420.405751768 # MHz

z_900=z_freq(nu_rest_21,900.)
r0_900=comoving_distance(z_900)
sigma_900=r0_900*np.tan(3.89*pi/180.)
print('consider the case where you have maximal sensitivity to the 21-cm signal at 900 MHz (z='+str(z_900)+')')
print('r0    = ',r0_900,'Mpc')
print('sigma = ',sigma_900,'Mpc')

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

def bias(F,B):
    return (np.linalg.inv(F)@B).reshape((F.shape[0],))

def printparswbiases(pars,parnames,biases):
    print('pars=',pars)
    print('parnames=',parnames)
    print('biases=',biases)
    for p,par in enumerate(pars):
        print('{:12} = {:-10.3e} with bias {:-12.5e}'.format(parnames[p], par, biases[p]))
    return None

nk=25
kvec=np.linspace(0.05,0.7,nk)
sigk=0.05*np.cos(2*np.pi*(kvec-kvec[0])/(kvec[-1]-kvec[0]))+0.01 # uncertainty in the power spectrum at each k-mode ... just a toy case sinusoidally more sensitive in the middle but offset vertically so there is a positive uncertainty at each k

CAMBpars=np.asarray([67.7,0.022,0.119,2.1e-9, 0.97])
CAMBparnames=['H_0','Omega_b h^2','Omega_c h^2','A_S','n_s']
CAMBparnames_LaTeX=['$H_0$','$\Omega_b h^2$','$\Omega_c h^2$','$A_S$','$n_s$']
CAMBpars[3]/=scale
nprm=len(CAMBpars) # number of parameters
CAMBdpar=1e-3*np.ones(nprm)
CAMBdpar[3]*=scale
ztest=7.4
CAMBk,CAMBPtrue=get_mps(CAMBpars,ztest,npts=nk)

plt.figure()
plt.semilogy(CAMBk,np.reshape(CAMBPtrue,(nk,)))
plt.xlabel('k (Mpc)')
plt.ylabel('P (K^2 Mpc^{-3})')
plt.title('z='+str(ztest)+' power spectrum')
plt.show()
assert(1==0)

# CAMBnpars=len(CAMBpars)
# calcCAMBPpartials=False

# npts=25
# rkmax=104.
# rk_test=np.linspace(0,rkmax,npts)
# sig_test=25
# r0_test=75
# W=W_binned_airy_beam(rk_test,sig_test,r0_test)
# epsvals=np.logspace(-15,0,20) # multiplicative prefactor: "what fractional error do you have in your knowledge of the beam width"

# for eps in epsvals:
#     print('\neps=',eps)
#     Wthought=W_binned_airy_beam(rk_test,sig_test,(1.+eps)*r0_test)

#     # CAMB MATTER POWER SPECTRUM CASE
#     plt.figure()
#     plt.imshow(W-Wthought)
#     plt.colorbar()
#     plt.title('W-Wthought check')
#     plt.show()
#     CAMBPcont=(W-Wthought)@CAMBPtrue.T
#     if calcCAMBPpartials:
#         CAMBPpartials=buildCAMBpartials(CAMBpars,ztest,nk,CAMBdpar) # buildCAMBpartials(p,z,nmodes,dpar)
#         np.save('cambppartials.npy',CAMBPpartials)
#     else:
#         CAMBPpartials=np.load('cambppartials.npy')
#     CAMBF=fisher(CAMBPpartials,sigk)
#     CAMBPcontDivsigk=(CAMBPcont.T/sigk).T
#     CAMBB=(CAMBPpartials.T@(CAMBPcontDivsigk))
#     CAMBb=bias(CAMBF,CAMBB)

#     CAMBpars2=CAMBpars.copy()
#     CAMBpars2[3]*=scale
#     CAMBb2=CAMBb.copy()
#     CAMBb2[3]*=scale
#     print('\nCAMB matter PS')
#     printparswbiases(CAMBpars2,CAMBparnames,CAMBb2)
#     assert(1==0)