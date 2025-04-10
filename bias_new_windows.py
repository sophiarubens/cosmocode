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

nu_ctr=900 # centre frequency of survey in MHz
z_ctr=freq2z(nu_rest_21,nu_ctr)
r0_ctr=comoving_distance(z_ctr)
survey_width=60. # survey bandwidth in MHz ... based on the 1/15 deltanu/nu ratio inspired by HERA cosmological surveys
nu_lo=nu_ctr-survey_width/2.
z_hi=freq2z(nu_rest_21,nu_lo)
Dc_hi=comoving_distance(z_hi)
nu_hi=nu_ctr+survey_width/2.
z_lo=freq2z(nu_rest_21,nu_hi)
Dc_lo=comoving_distance(z_lo)
deltaz=z_hi-z_lo
N_CHORDcosmo=2048.
channel_width=survey_width/N_CHORDcosmo # channel width in MHz
surv_channels=np.arange(nu_lo,nu_hi-channel_width,channel_width)
print("survey centred at",nu_ctr,"MHz / z=",z_ctr,"/ D_c=",r0_ctr,"Mpc")
print("survey spans",nu_lo,"-",nu_hi,"MHz (width=",survey_width,"MHz) in",N_CHORDcosmo,"channels of width",channel_width,"MHz")
print("or, in redshift space, z=",z_hi,"-",z_lo,"(deltaz=",deltaz,")")
print("or, in comoving distance terms, D_c=",Dc_hi,"-",Dc_lo,"Mpc")

sig_LoS=0.5*(Dc_hi-Dc_lo) # of course this flattens the nonlinearity of Dc(z) and ignores the asymmetry in sensitivity WRT the centre
print("sig_LoS=",sig_LoS,"Mpc")

rk_surv=kpar(surv_channels,N_CHORDcosmo)
deltakpar_initial=rk_surv[1]-rk_surv[0]
deltakpar_final  =rk_surv[-1]-rk_surv[-2]
deltadeltakpar=deltakpar_final-deltakpar_initial
print("deltadeltakpar=deltakpar_final-deltakpar_initial=",deltadeltakpar)
print("deltadeltakpar/deltakpar_initial=",deltadeltakpar/deltakpar_initial)
print("deltadeltakpar/deltakpar_final  =",deltadeltakpar/deltakpar_final)
linearized_kbins=np.arange(rk_surv[0],rk_surv[0]+N_CHORDcosmo*deltakpar_initial,deltakpar_initial)

# plt.figure()
# plt.plot(rk_surv,label="full nonlin version")
# plt.plot(linearized_kbins,label="linear appx w/ init deltakpar")
# plt.xlabel("bin number")
# plt.ylabel("bin floor (Mpc$^{-1}$)")
# plt.title("reality check for k-bin spacing")
# plt.legend()
# plt.tight_layout()
# plt.savefig("check_k_bin_spacing.png")
# plt.show()
# assert(1==0), "examining differential k-bin width"

# STILL USING A TOY MODEL FOR 1D k-bin variance
sigk_cos_ampl=1e-7 
sigk_cos_offs=5e-7 # choose an offs>ampl so sigk remains positive everywhere
k_bin_stddev=sigk_cos_ampl*np.cos(2*np.pi*(rk_surv-rk_surv[0])/(rk_surv[-1]-rk_surv[0]))+sigk_cos_offs # even worse now b/c I hope to use the nonlinear k-bins
print('sigk_cos_ampl=',sigk_cos_ampl)
print('sigk_cos_offs=',sigk_cos_offs)
assert(1==0), "consider adding 21cmSense 1D variances"

CAMBpars=np.asarray([ H0_Planck18, Omegabh2_Planck18,  Omegach2_Planck18,  AS_Planck18,  ns_Planck18])
CAMBparnames=       ['H_0',       'Omega_b h^2',      'Omega_c h^2',      'A_S',        'n_s'       ]
CAMBparnames_LaTeX= ['$H_0$',     '$\Omega_b h^2$',   '$\Omega_c h^2$',   '$A_S$',      '$n_s$'     ]
CAMBpars[3]/=scale
CAMBnpars=len(CAMBpars)
calcCAMBPpartials=True
nprm=len(CAMBpars) # number of parameters
CAMBdpar=1e-3*np.ones(nprm)
CAMBdpar[3]*=scale
CAMBk,CAMBPtrue=get_mps(CAMBpars,z_ctr,npts=N_CHORDcosmo)

npts=2222
theta_vals=np.linspace(0,twopi,npts)
basic_airy_beam=(j1(theta_vals)/theta_vals)**2
basic_airy_beam_half_max=1./8. # derived on paper
beta_fwhm=theta_vals[np.nanargmin(np.abs(basic_airy_beam-basic_airy_beam_half_max))]
CHORD_ish_fwhm=pi/45. # 4 deg = 4pi/180 rad = pi/45 rad
CHORD_ish_airy_alpha=beta_fwhm/CHORD_ish_fwhm
Wrscipy=  W_binned_airy_beam(rk_surv,sig_LoS,r0_ctr,CHORD_ish_airy_alpha,'scipy') # W_binned_airy_beam(rk_vector,sig,r0,alpha,r_like_strategy,save=False,verbose=False)
Wrhand=   W_binned_airy_beam(rk_surv,sig_LoS,r0_ctr,CHORD_ish_airy_alpha,'hand')
Wrwiggly= W_binned_airy_beam(rk_surv,sig_LoS,r0_ctr,CHORD_ish_airy_alpha,'wiggly')
assert(1==0), "checking that the new args= and alpha call structure works in the Wbinned calc"

fig,axs=plt.subplots(1,3,figsize=(20,5))
axs[0].plot(rk_surv,Wrscipy[0])
axs[0].set_title("W[0] (scipy quad r-like term)")
axs[1].plot(rk_surv,Wrhand[0])
axs[1].set_title("W[0] (hand-calculated r-like term)")
axs[2].plot(rk_surv,Wrwiggly[0])
axs[2].set_title("W[0] (wiggly version of hand-calculated r-like term)")
for ax in axs:
    ax.set_xlabel("k (Mpc$^{-1})")
    ax.set_ylabel("Windowing amplitude (dimensionless)")
plt.tight_layout()
plt.savefig("separate_window_calc_strategy_comparison.png")
plt.show()

# fig,axs=plt.subplots(1,2)
# epsvals=np.logspace(-6,-0.4,9) # multiplicative prefactor: "what fractional error do you have in your knowledge of the beam width"
# fih,axh=plt.subplots(3,3,figsize=(10,10),layout='tight')

# for k,eps in enumerate(epsvals):
#     i=k//3
#     j=k%3
#     print('\neps=',eps)
#     # Wthought=W_binned_airy_beam_r_hand(rk_surv,sig_LoS,r0_ctr,alpha=(1+eps)*CHORD_ish_airy_alpha)
#     Wthought= W_binned_airy_beam(rk_surv,sig_LoS,r0_ctr,CHORD_ish_airy_alpha,'wiggly')

#     im=axh[i,j].imshow(W-Wthought)
#     plt.colorbar(im,ax=axh[i,j])
#     axh[i,j].set_xlabel("k")
#     axh[i,j].set_ylabel("k'")
#     axh[i,j].set_title("eps="+str(eps))

#     if calcCAMBPpartials:
#         CAMBPpartials=buildCAMBpartials(CAMBpars,ztest,N_CHORDcosmo,CAMBdpar)
#         np.save('cambppartials.npy',CAMBPpartials)
#     else:
#         CAMBPpartials=np.load('cambppartials.npy')

#     CAMBPcont=(W-Wthought)@CAMBPtrue.T
#     CAMBF=fisher(CAMBPpartials,k_bin_stddev)
#     CAMBPcontDivsigk=(CAMBPcont.T/k_bin_stddev).T
#     CAMBB=(CAMBPpartials.T@(CAMBPcontDivsigk))
#     CAMBb=bias(CAMBF,CAMBB)

#     CAMBpars2=CAMBpars.copy()
#     CAMBpars2[3]*=scale
#     CAMBb2=CAMBb.copy()
#     CAMBb2[3]*=scale
#     print('\nCAMB matter PS **R-LIKE BY HAND**')
#     printparswbiases(CAMBpars2,CAMBparnames,CAMBb2)
# fih.suptitle('W-Wthought for various fractional errors in beam width R HAND')
# fih.savefig('W_minus_Wthought_beam_width_tests_R_HAND.png')
# fih.show()