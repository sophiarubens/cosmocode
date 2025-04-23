import numpy as np
from matplotlib import pyplot as plt
# import pygtc
# import camb
# from camb import model
# from calculate_airy_gaussian_window import *
from cyl_bin_window import *
from cosmo_distances import *
from bias_helper_fcns import *

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
scale=1e-9

nu_ctr=900 # centre frequency of survey in MHz
z_ctr=freq2z(nu_rest_21,nu_ctr)
Dc_ctr=comoving_distance(z_ctr)
survey_width=60. # survey bandwidth in MHz ... based on the 1/15 deltanu/nu ratio inspired by HERA cosmological surveys
nu_lo=nu_ctr-survey_width/2.
z_hi=freq2z(nu_rest_21,nu_lo)
Dc_hi=comoving_distance(z_hi)
nu_hi=nu_ctr+survey_width/2.
z_lo=freq2z(nu_rest_21,nu_hi)
Dc_lo=comoving_distance(z_lo)
deltaz=z_hi-z_lo
channel_width=0.183 # 183 kHz from CHORD Wiki -> SWGs -> Galaxies -> Spectral resolution
N_CHORDcosmo=survey_width/channel_width
N_CHORDcosmo_int=int(N_CHORDcosmo)
surv_channels=np.arange(nu_lo,nu_hi,channel_width)
print("survey centred at",nu_ctr,"MHz / z=",z_ctr,"/ D_c=",Dc_ctr,"Mpc")
print("survey spans",nu_lo,"-",nu_hi,"MHz (width=",survey_width,"MHz) in",N_CHORDcosmo_int,"channels of width",channel_width,"MHz")
print("or, in redshift space, z=",z_hi,"-",z_lo,"(deltaz=",deltaz,")")
print("or, in comoving distance terms, D_c=",Dc_hi,"-",Dc_lo,"Mpc")

sig_LoS=0.25*(Dc_ctr-Dc_lo)/10 # dialing in the bound set by condition following from linearization...
print("sig_LoS=",sig_LoS,"Mpc")

kpar_surv=kpar(nu_ctr,channel_width,N_CHORDcosmo_int) # kpar(nu_ctr,chan_width,N_chan,H0=H0_Planck18)
print("kpar_surv check: kparmin,kparmax=",kpar_surv[0],kpar_surv[-1])

N_CHORDbaselines=1010 # upper bound (b/c not sure if the grid gaps will remove redundance or just unique baselines) from my formula is 1010
b_NS_CHORD=8.5 # m
N_NS_CHORD=24
b_EW_CHORD=6.3 # m
N_EW_CHORD=22
bminCHORD=6.3
bmaxCHORD=np.sqrt((b_NS_CHORD*N_NS_CHORD)**2+(b_EW_CHORD*N_EW_CHORD)**2) # too optimistic ... this is a low-redundancy baseline and the numerics will be better if I don't insist upon being so literal
# bmaxCHORD=b_NS_CHORD*N_NS_CHORD
# bmaxCHORD=b_EW_CHORD*N_EW_CHORD
kperp_surv=kperp(nu_ctr,N_CHORDbaselines,bminCHORD,bmaxCHORD) # kperp(nu_ctr,N_modes,bmin,bmax)
print("kperp_surv check: kperpmin,kperpmax=",kperp_surv[0],kperp_surv[-1])

CHORD_ish_fwhm_surv=(1./12.)*pi/180. # CHORD pathfinder spec page:
# D3A6 beam measurements in Ian's MSc thesis, taken by inspection from the plot and eyeball-averaged over the x- and y-pols 4 deg = 4pi/180 rad = pi/45 rad # approximate, but specific to this hypothetical 900 MHz survey 
btype="gaussian" 
W_surv=False
W_surv=   W_cyl_binned(kpar_surv,kperp_surv,sig_LoS,Dc_ctr,btype,CHORD_ish_fwhm_surv, savename="survey") # W_cyl_binned(kparvec,kthetavec,sigLoS,beamtype,thetaHWHM,save=False)
# n_inspect=2048
# kperp_inspect=np.linspace(0,0.1,1010) # I don't seem to be looking at a different regime than the interesting one with my survey calculation ... so might as well recycle it
# W_inspect=W_cyl_binned(kpar_surv,kperp_inspect,sig_LoS,Dc_ctr,btype,CHORD_ish_fwhm_surv, savename="inspect")
assert(1==0), "checking W_cyl_binned procedure using _inspect k-modes"

# STILL USING A TOY MODEL FOR 1D k-bin variance
sigk_cos_ampl=0.1 
sigk_cos_offs=0.5 # choose an offs>ampl so sigk remains positive everywhere
k_bin_stddev=sigk_cos_ampl*np.cos(2*np.pi*(kpar_surv-kpar_surv[0])/(kpar_surv[-1]-kpar_surv[0]))+sigk_cos_offs # even worse now b/c I hope to use the nonlinear k-bins
print('sigk_cos_ampl=',sigk_cos_ampl)
print('sigk_cos_offs=',sigk_cos_offs)

# ##### ##### 1D 21cmSense example with the default z=9.5 matter PS baked in
# k_bins_21cmse_knows_about=np.load("chord_21cmse_k1d.npy")
# # print("k_bins_21cmse_knows_about==kpar_surv",k_bins_21cmse_knows_about==kpar_surv)
# k_bin_stddev_21cmse=np.load("chord_21cmse_sigk.npy")
# print("k_bin_stddev_21cmse.shape",k_bin_stddev_21cmse.shape)
# print("k_bin_stddev.shape=",k_bin_stddev.shape)
# ##### #####

##

##

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

fig,axs=plt.subplots(1,2,figsize=(15,5))
im=axs[0].imshow(W_surv,extent=[kperp_surv[0],kperp_surv[-1],kpar_surv[-1],kpar_surv[0]],aspect=kperp_surv[-1]/kpar_surv[-1])
axs[0].set_title("k-modes for such a survey")
fig.colorbar(im,ax=axs[0],label="Dimensionless windowing amplitude")
im=axs[1].imshow(W_inspect,extent=[kperp_inspect[0],kperp_inspect[-1],kpar_inspect[-1],kpar_inspect[0]],aspect=kperp_inspect[-1]/kpar_inspect[-1])
fig.colorbar(im,ax=axs[1],label="Dimensionless windowing amplitude")
axs[1].set_title("inset (beyond the survey) to check shape intuition")
for ax in axs:
    ax.set_xlabel("k$_{\perp}$ (Mpc$^{-1}$)")
    ax.set_ylabel("k$_{||}$ (Mpc$^{-1}$)")
plt.suptitle("Wbinned[k,k'=0] with instrument response parameters for a 900 MHz CHORD-like survey")
plt.tight_layout()
plt.savefig("binned_window_inspection.png")
plt.show()
# print("2*(FWHM_(FTed)/(2sqrt(2ln2)))**2=",2*(0.004959)/(2*np.sqrt(2*np.log(2))))
# print("1/sig_LoS=",1/sig_LoS)
# print("2*np.sqrt(2*np.log(2))/sig_LoS=",2*np.sqrt(2*np.log(2))/sig_LoS)
assert(1==0), "checking cylindrically binned windows"

epsvals=np.logspace(-6,-0.4,9) # multiplicative prefactor: "what fractional error do you have in your knowledge of x response parameter"
fih,axh=plt.subplots(3,3,figsize=(10,10),layout='tight')

W=W_surv # <<<<<<<<<<<<
for k,eps in enumerate(epsvals):
    i=k//3
    j=k%3
    print('\neps=',eps)
    Wthought=W_binned(kpar_surv,(1+eps)*sig_LoS,Dc_ctr,CHORD_ish_fwhm_surv,'wiggly',btype) # <<<<<<<<<<<<

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