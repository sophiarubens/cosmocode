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
c=2.998e8
pc=30856775814914000 # m
Mpc=pc*1e6

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
    for p,par in enumerate(pars):
        print('{:12} = {:-10.3e} with bias {:-12.5e}'.format(parnames[p], par, biases[p]))
    return None

z_900=freq2z(nu_rest_21,900.)
r0_900=comoving_distance(z_900)
r0_800=comoving_distance(freq2z(nu_rest_21,800.))
r0_1000=comoving_distance(freq2z(nu_rest_21,1000.))
print("r0_800,r0_900,r0_1000=",r0_800,r0_900,r0_1000,"Mpc")
print("r0_800,r0_900,r0_1000=",r0_800*Mpc,r0_900*Mpc,r0_1000*Mpc,"m")
sig_LoS_900=0.5*(r0_800-r0_1000) # placeholder value ... inspired by the CHIME 21 cm survey being about 200 MHz wide -> take the half-width of the comoving distance range corresp. to this interval
print("sig_LoS_900=",sig_LoS_900)
b_min_CHORD=6.3 # m
b_max_CHORD=245

# kpar(surv_channels,N)

# kperpmin_minv,kperpmax_minv=kperplims(900.,b_min_CHORD,b_max_CHORD) # kperplims(nuctr,bmin,bmax,nurest=nu21):# here, use bmin as also the limiting deltab -> use to set deltak
# print('m^{-1}: kperpmin,kperpmax=',kperpmin_minv,kperpmax_minv)
# kperpmin=kperpmin_minv*Mpc # in Mpcinv
# kperpmax=kperpmax_minv*Mpc
# print('Mpc^{-1}: kperpmin,kperpmax=',kperpmin,kperpmax)
# rk_900=np.arange(kperpmin,kperpmax,kperpmin)

# deltakpar(nuctr,deltanu,nurest=nu21)
deltanu_900=60./2048. # MHz ... based on the 1/15 deltanu/nu ratio inspired by HERA cosmological surveys
# deltakpar_900=deltakpar(900.,deltanu_900)
# print("deltakpar_900=",deltakpar_900)

# assert(1==0), "checking k-limit calculation"
# kparmin_m=twopi*800e6/c # m
# kparmax_m=twopi*1000e6/c
# kparmin= # Mpc

### DELTA_NU = 0.195 MHZ !!!!!!!!!!!!!!! FROM TELECON SLIDES!!! Δν = 0.195 MHz >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> PICK UP HERE TMRW!! (although quoted as 30 kHz in an earlier slide... that is an older number) (even older: 8192 frequency bins (183 kHz resolution) (32k channels ... before that) ... varies by project; check the 22-11-09 slides!!!
nk=len(rk_900)
print('nk=',nk)

# STILL USING A TOY MODEL FOR 1D k-bin variance
# choose an offs>ampl so sigk remains positive everywhere
sigk_cos_ampl=1e-7
sigk_cos_offs=5e-7
sigk_900=sigk_cos_ampl*np.cos(2*np.pi*(rk_900-rk_900[0])/(rk_900[-1]-rk_900[0]))+sigk_cos_offs
print('sigk_cos_ampl=',sigk_cos_ampl)
print('sigk_cos_offs=',sigk_cos_offs)
print('consider the case where you have maximal sensitivity to the 21-cm signal at 900 MHz (z='+str(z_900)+')')
print('r0    = ',r0_900,'Mpc')
print('sigma_LoS = ',sig_LoS_900,'Mpc')

CAMBpars=np.asarray([67.7,0.022,0.119,2.1e-9, 0.97])
CAMBparnames=['H_0','Omega_b h^2','Omega_c h^2','A_S','n_s']
CAMBparnames_LaTeX=['$H_0$','$\Omega_b h^2$','$\Omega_c h^2$','$A_S$','$n_s$']
CAMBpars[3]/=scale
CAMBnpars=len(CAMBpars)
calcCAMBPpartials=False
nprm=len(CAMBpars) # number of parameters
CAMBdpar=1e-3*np.ones(nprm)
CAMBdpar[3]*=scale
ztest=7.4
CAMBk,CAMBPtrue=get_mps(CAMBpars,ztest,npts=nk)

Wrscipy=      W_binned_airy_beam(rk_900,sig_LoS_900,r0_900)
Wrhand=       W_binned_airy_beam_r_hand(rk_900,sig_LoS_900,r0_900)
Wrhandwiggly= W_binned_airy_beam_r_hand_wiggly(rk_900,sig_LoS_900,r0_900)

fig,axs=plt.subplots(1,3,figsize=(20,5))
axs[0].plot(rk_900,Wrscipy[0])
axs[0].set_title("W[0] (scipy quad r-like term)")
axs[1].plot(rk_900,Wrhand[0])
axs[1].set_title("W[0] (hand-calculated r-like term)")
axs[2].plot(rk_900,Wrhandwiggly[0])
axs[2].set_title("W[0] (wiggly version of hand-calculated r-like term)")
for ax in axs:
    ax.set_xlabel("k (Mpc$^{-1})")
    ax.set_ylabel("Windowing amplitude (dimensionless)")
plt.tight_layout()
plt.savefig("separate_window_calc_strategy_comparison.png")
plt.show()

# plt.figure()
# # plt.plot(rk_900,Wrscipy[0],      label="W[0] (scipy quad r-like term)")
# plt.plot(rk_900,Wrhand[0],       label="W[0] (hand-calculated r-like term)")
# plt.plot(rk_900,Wrhandwiggly[0], label="W[0] (wiggly version of hand-calculated r-like term)")
# plt.xlabel("k (Mpc$^{-1})")
# plt.ylabel("Windowing amplitude (dimensionless)")
# plt.legend()
# plt.savefig("superimposed_window_calc_strategy_comparison.png")
# plt.show()

# plt.figure()
# plt.plot(rk_900,Wrhand[0]-Wrhandwiggly[0],       label="W[0] (hand-calculated r-like term, nonwiggly-wiggly)")
# plt.xlabel("k (Mpc$^{-1}$)")
# plt.ylabel("Windowing amplitude (dimensionless)")
# plt.legend()
# plt.savefig("superimposed_window_calc_strategy_comparison.png")
# plt.show()

# fig,axs=plt.subplots(1,2)
# epsvals=np.logspace(-6,-0.4,9) # multiplicative prefactor: "what fractional error do you have in your knowledge of the beam width"
# fih,axh=plt.subplots(3,3,figsize=(10,10),layout='tight')

# for k,eps in enumerate(epsvals):
#     i=k//3
#     j=k%3
#     print('\neps=',eps)
#     Wthoughtrhand=W_binned_airy_beam_r_hand(rk_900,(1.+eps)*sig_LoS_900,r0_900)

#     im=axh[i,j].imshow(Wrhand-Wthoughtrhand)
#     plt.colorbar(im,ax=axh[i,j])
#     axh[i,j].set_xlabel("k")
#     axh[i,j].set_ylabel("k'")
#     axh[i,j].set_title("eps="+str(eps))

#     if calcCAMBPpartials:
#         CAMBPpartials=buildCAMBpartials(CAMBpars,ztest,nk,CAMBdpar)
#         np.save('cambppartials.npy',CAMBPpartials)
#     else:
#         CAMBPpartials=np.load('cambppartials.npy')

#     CAMBPcontrhand=(Wrhand-Wthoughtrhand)@CAMBPtrue.T
#     CAMBFrhand=fisher(CAMBPpartials,sigk_900)
#     CAMBPcontDivsigkrhand=(CAMBPcontrhand.T/sigk_900).T
#     CAMBBrhand=(CAMBPpartials.T@(CAMBPcontDivsigkrhand))
#     CAMBbrhand=bias(CAMBFrhand,CAMBBrhand)

#     CAMBpars2=CAMBpars.copy()
#     CAMBpars2[3]*=scale
#     CAMBb2rhand=CAMBbrhand.copy()
#     CAMBb2rhand[3]*=scale
#     print('\nCAMB matter PS **R-LIKE BY HAND**')
#     printparswbiases(CAMBpars2,CAMBparnames,CAMBb2rhand)
# fih.suptitle('W-Wthought for various fractional errors in beam width R HAND')
# fih.savefig('W_minus_Wthought_beam_width_tests_R_HAND.png')
# fih.show()