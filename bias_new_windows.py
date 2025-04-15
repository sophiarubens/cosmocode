import sys
sys.path.append('/Users/sophiarubens/Downloads/research/code/param_bias/')
import numpy as np
from matplotlib import pyplot as plt
# import pygtc
from scipy.special import j1 # first-order Bessel function of the first kind
from scipy.integrate import quad,dblquad
import camb
from camb import model
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
N_CHORDcosmo_int=int(N_CHORDcosmo)
channel_width=survey_width/N_CHORDcosmo # channel width in MHz
surv_channels=np.arange(nu_lo,nu_hi,channel_width)
print("survey centred at",nu_ctr,"MHz / z=",z_ctr,"/ D_c=",r0_ctr,"Mpc")
print("survey spans",nu_lo,"-",nu_hi,"MHz (width=",survey_width,"MHz) in",N_CHORDcosmo_int,"channels of width",channel_width,"MHz")
print("or, in redshift space, z=",z_hi,"-",z_lo,"(deltaz=",deltaz,")")
print("or, in comoving distance terms, D_c=",Dc_hi,"-",Dc_lo,"Mpc")

sig_LoS=0.5*(Dc_hi-Dc_lo) # of course this flattens the nonlinearity of Dc(z) and ignores the asymmetry in sensitivity WRT the centre
print("sig_LoS=",sig_LoS,"Mpc")

rk_surv=kpar(nu_ctr,channel_width,N_CHORDcosmo_int) # kpar(nu_ctr,chan_width,N_chan,H0=H0_Planck18)
print("rk_surv check: kparmin,kparmax=",rk_surv[0],rk_surv[-1])

# STILL USING A TOY MODEL FOR 1D k-bin variance
sigk_cos_ampl=0.1 
sigk_cos_offs=0.5 # choose an offs>ampl so sigk remains positive everywhere
k_bin_stddev=sigk_cos_ampl*np.cos(2*np.pi*(rk_surv-rk_surv[0])/(rk_surv[-1]-rk_surv[0]))+sigk_cos_offs # even worse now b/c I hope to use the nonlinear k-bins
print('sigk_cos_ampl=',sigk_cos_ampl)
print('sigk_cos_offs=',sigk_cos_offs)

# ##### #####
# k_bins_21cmse_knows_about=np.load("chord_21cmse_k1d.npy")
# # print("k_bins_21cmse_knows_about==rk_surv",k_bins_21cmse_knows_about==rk_surv)
# k_bin_stddev_21cmse=np.load("chord_21cmse_sigk.npy")
# print("k_bin_stddev_21cmse.shape",k_bin_stddev_21cmse.shape)
# print("k_bin_stddev.shape=",k_bin_stddev.shape)
# assert(1==0), "can't use 21cmSense versions until I fix the redshift specification there"
# ##### #####

CAMBpars=np.asarray([ H0_Planck18, Omegabh2_Planck18,  Omegach2_Planck18,  AS_Planck18,  ns_Planck18])
CAMBparnames=       ['H_0',       'Omega_b h^2',      'Omega_c h^2',      'A_S',        'n_s'       ]
CAMBparnames_LaTeX= ['$H_0$',     '$\Omega_b h^2$',   '$\Omega_c h^2$',   '$A_S$',      '$n_s$'     ]
CAMBpars[3]/=scale
CAMBnpars=len(CAMBpars)
calcCAMBPpartials=False
nprm=len(CAMBpars) # number of parameters
CAMBdpar=1e-3*np.ones(nprm)
CAMBdpar[3]*=scale
CAMBk,CAMBPtrue=get_mps(CAMBpars,z_ctr,npts=int(N_CHORDcosmo))
CAMBsave=False
if CAMBsave:
    np.save("camb_k.npy",CAMBk)
    np.save("camb_P.npy",CAMBPtrue)

if calcCAMBPpartials:
    CAMBPpartials=buildCAMBpartials(CAMBpars,z_ctr,N_CHORDcosmo_int,CAMBdpar)
    np.save('cambppartials.npy',CAMBPpartials)
else:
    CAMBPpartials=np.load('cambppartials.npy')

CHORD_ish_fwhm_surv=pi/45. # 4 deg = 4pi/180 rad = pi/45 rad # approximate, but specific to this hypothetical 900 MHz survey 
btype="arbitrary" 
W_surv=   W_binned(rk_surv,   sig_LoS,r0_ctr,CHORD_ish_fwhm_surv,'wiggly', btype)
rk_inspect=np.linspace(0,0.02,N_CHORDcosmo_int)
W_inspect=W_binned(rk_inspect,sig_LoS,r0_ctr,CHORD_ish_fwhm_surv,'wiggly', btype)

# print("START OF NORMALIZATION CHECK USING WRHAND")
# print("np.sum(Wrhand)=",np.sum(Wrhand))
# print("np.sum(Wrhand[0])=",np.sum(Wrhand[0]))
# print("END OF NORMALIZATION CHECK USING WRHAND")

fig,axs=plt.subplots(1,2,figsize=(15,5))
axs[0].plot(rk_surv,    W_surv[0])
axs[0].set_title("k-modes for such a survey")
axs[1].plot(rk_inspect, W_inspect[0])
axs[1].set_title("lower-k inset (beyond the survey) to check shape intuition")
for ax in axs:
    ax.set_xlabel("k (Mpc$^{-1}$)")
    ax.set_ylabel("Normalized windowing amplitude (dimensionless)")
plt.suptitle("Wbinned[k,k'=0] with instrument response parameters for a 900 MHz CHORD-like survey")
plt.tight_layout()
plt.savefig("binned_window_inspection.png")
plt.show()
# print("0.5*(FWHM_(FTed)/(2sqrt(2ln2)))**2=",0.5*(0.004959)/(2*np.sqrt(2*np.log(2))))
# print("(FWHM_(FTed)/(2sqrt(2ln2)))**2=",0.004959/(2*np.sqrt(2*np.log(2))))
print("2*(FWHM_(FTed)/(2sqrt(2ln2)))**2=",2*(0.004959)/(2*np.sqrt(2*np.log(2))))
print("1/sig_LoS=",1/sig_LoS)
print("2*np.sqrt(2*np.log(2))/sig_LoS=",2*np.sqrt(2*np.log(2))/sig_LoS)

epsvals=np.logspace(-6,-0.4,9) # multiplicative prefactor: "what fractional error do you have in your knowledge of x response parameter"
fih,axh=plt.subplots(3,3,figsize=(10,10),layout='tight')

W=W_surv # <<<<<<<<<<<<
for k,eps in enumerate(epsvals):
    i=k//3
    j=k%3
    print('\neps=',eps)
    Wthought=W_binned(rk_surv,(1+eps)*sig_LoS,r0_ctr,CHORD_ish_fwhm_surv,'wiggly',btype) # <<<<<<<<<<<<

    im=axh[i,j].imshow(W-Wthought)
    plt.colorbar(im,ax=axh[i,j])
    axh[i,j].set_xlabel("k")
    axh[i,j].set_ylabel("k'")
    axh[i,j].set_title("eps="+str(eps))

    CAMBPcont=(W-Wthought)@CAMBPtrue.T
    CAMBF=fisher(CAMBPpartials,k_bin_stddev)
    CAMBPcontDivsigk=(CAMBPcont.T/k_bin_stddev).T
    CAMBB=(CAMBPpartials.T@(CAMBPcontDivsigk))
    CAMBb=bias(CAMBF,CAMBB)

    CAMBpars2=CAMBpars.copy()
    CAMBpars2[3]*=scale
    CAMBb2=CAMBb.copy()
    CAMBb2[3]*=scale
    print('\nCAMB matter PS **R-LIKE BY HAND**')
    printparswbiases(CAMBpars2,CAMBparnames,CAMBb2)
    assert(1==0),"debbugging unreasonable orders of magnitude"
fih.suptitle('W-Wthought for various fractional errors in beam width R HAND')
fih.savefig('W_minus_Wthought_beam_width_tests_R_HAND.png')
fih.show()