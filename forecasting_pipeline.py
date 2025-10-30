import numpy as np
import camb
from camb import model
from scipy.signal import convolve
from scipy.interpolate import interpn,interp1d
from scipy.special import j1
from numpy.fft import fftshift,ifftshift,fftn,irfftn,fftfreq,ifft2
from cosmo_distances import *
from matplotlib import pyplot as plt
import matplotlib
import time
import scipy.sparse as spsp

# cosmological
Omegam_Planck18=0.3158
Omegabh2_Planck18=0.022383
Omegach2_Planck18=0.12011
OmegaLambda_Planck18=0.6842
lntentenAS_Planck18=3.0448
tentenAS_Planck18=np.exp(lntentenAS_Planck18)
AS_Planck18=tentenAS_Planck18/10**10
ns_Planck18=0.96605
H0_Planck18=67.32
h_Planck18=H0_Planck18/100.
Omegamh2_Planck18=Omegam_Planck18*h_Planck18**2
pars_set_cosmo_Planck18=[H0_Planck18,Omegabh2_Planck18,Omegamh2_Planck18,AS_Planck18,ns_Planck18] # suitable for get_mps

# physical
nu_HI_z0=1420.405751768 # MHz
c=2.998e8

# mathematical
pi=np.pi
twopi=2.*pi
ln2=np.log(2)

# computational
infty=np.inf # np.infty deprecated in numpy 2.0
maxfloat= np.finfo(np.float64).max
huge=np.sqrt(maxfloat)
maxfloat= np.finfo(np.float64).max
maxint=   np.iinfo(np.int64  ).max
nearly_zero=(1./maxfloat)**2

# numerical
scale=1e-9
BasicAiryHWHM=1.616339948310703178119139753683896309743121097215461023581 # a preposterous number of sig figs from Mathematica (I haven't counted them but this is almost certainly overkill/ past the double-precision threshold)
eps=1e-15

# CHORD layout
N_NS_full=24
N_EW_full=22
DRAO_lat=49.320791*np.pi/180. # Google Maps satellite view, eyeballing what looks like the middle of the CHORD site: 49.320791, -119.621842 (bc considering drift-scan CHIME-like "pointing at zenith" mode, same as dec)
D=6. # m

class NotYetImplementedError(Exception):
    pass
class NumericalDeltaError(Exception):
    pass
class ResolutionError(Exception):
    pass
class UnsupportedBinningMode(Exception):
    pass
class NotEnoughInfoError(Exception):
    pass
class PathologicalError(Exception):
    pass
class ConflictingInfoError(Exception):
    pass
class SurveyOutOfBoundsError(Exception): # make these inherit from each other (or something) to avoid repetitive code
    pass

def extrapolation_warning(regime,want,have):
    print("WARNING: if extrapolation is permitted in the interpolate_P call, it will be conducted for {:15s} (want {:9.4}, have{:9.4})".format(regime,want,have))
    return None
def represent(cosmo_stats_instance):
    attributes=vars(cosmo_stats_instance)
    representation='\n'.join("%s: %s" % item for item in attributes.items())
    print(representation)

def UAA_Gaussian(X,Y,fwhm_x,fwhm_y,r0):
    """
    (Nvox,Nvox,Nvox) Cartesian box (z=LoS direction), centred at r0, sampling the response fcn at each point
    """
    return np.exp(-ln2*((X/fwhm_x)**2+(Y/fwhm_y)**2)/r0**2)

def UAA_Airy(X,Y,fwhm_x,fwhm_y,r0):
    """
    (Nvox,Nvox,Nvox) Cartesian box (z=LoS direction), centred at r0, sampling the response fcn at each point
    """
    thetaX=X/r0
    argX=thetaX*BasicAiryHWHM/fwhm_x
    thetaY=Y/r0
    argY=thetaY*BasicAiryHWHM/fwhm_y
    perp=((j1(argX+eps)*j1(argY+eps))/((argX+eps)*(argY+eps)))**2
    return perp

"""
this class helps compute contaminant power and cosmological parameter biases using
a Fisher-based formalism using two complementary strategies with different scopes:
1. analytical windowing for a cylindrically symmetric Gaussian beam
2. numerical  windowing for a Gaussian beam with different x- and y-pol widths
"""

class beam_effects(object):
    def __init__(self,
                 bmin,bmax,                                             # extreme baselines of the array
                 ceil,                                                  # avoid kpars beyond the regime of linear theory
                 primary_beam_categ,primary_beam_type,                  # primary beam considerations
                 primary_beam_aux,primary_beam_uncs,                    # primary beam details
                 pars_set_cosmo,pars_forecast,                          # implement soon: build out the functionality for pars_forecast to differ nontrivially from pars_set_cosmo
                 n_sph_modes,dpar,                                      # conditioning the CAMB/etc. call
                 nu_ctr,delta_nu,                                       # for the survey of interest
                 evol_restriction_threshold=1./15.,                     # misc. numerical considerations
                 init_and_box_tol=0.05,CAMB_tol=0.05,                   # considerations for k-modes at different steps
                 ftol_deriv=1e-6,eps=1e-16,maxiter=5,                   # precision control for numerical derivatives
                 uncs=None,frac_unc=0.1,                                # for Fisher-type calcs
                 Nkpar_box=15,Nkperp_box=18,frac_tol_conv=0.1,          # considerations for cyl binned power spectra from boxes
                 pars_forecast_names=None,                              # for verbose output
                 manual_primary_beam_modes=None,                        # config space pts at which a pre–discretely sampled primary beam is known
                 no_monopole=True):                                     # to implement: dirty image stacking compatibility, other primary beam types, and more
        """
        bmin,bmax                  :: floats                       :: max and min baselines of the array       :: m
        ceil                       :: int                          :: # high-kpar channels to ignore           :: ---
        primary_beam_categ         :: str                          :: * UAA = uniform across the array         :: ---
                                                                      * PA  = per-antenna
                                                                      * manual = pathological from elsewhere
        primary_beam_type          :: str                          :: * UAA: Gaussian, Airy                    :: ---
                                                                      * PA: Gaussian [MORE IN PROGRESS]
                                                                      * manual: None
        primary_beam_aux           :: (N_args,) of floats          :: * UAA:pass FWHMs; r0 appended internally :: r0:           Mpc
                                                                      * manual: primary beams evaluated on the    fwhms:        rad
                                                                        grid of interest, a list ordered as       evaled beams: ---
                                                                        [fidu,pert]
                                                                      * PA-internal: FWHMs  
        primary_beam_uncs          :: (2,) of floats               :: fractional uncertainties for x and y     :: ---
        pars_set_cosmo             :: (N_fid_pars,) of floats      :: params to condition a CAMB/etc. call     :: as found in ΛCDM
        pars_forecast              :: (N_forecast_pars,) of floats :: params for which you'd like to forecast  :: as found in ΛCDM
        n_sph_modes                :: int                          :: # modes to put in CAMB/etc. MPS          :: ---
        dpar                       :: (N_forecast_pars,) of floats :: initial guess of num. dif. step sizes    :: same as for pars_forecast
        nu_ctr                     :: float                        :: central freq for survey of interest      :: MHz
        delta_nu                   :: float                        :: channel width for survey of interest     :: MHz
        evol_restriction_threshold :: float                        :: ~$\frac{\Delta z}{z}$ w/in survey box    :: ---
        init_and_box_tol, CAMB_tol :: floats                       :: how much wider do you want the k-ranges  :: ---
                                                                      of preceding steps to be? (frac tols)
        ftol_deriv                 :: float                        :: frac tol relating to scale of fcn range  :: ---
        eps                        :: float                        :: tiny offset factor to protect against    :: --- 
                                                                      numerical division-by-zero errors
        maxiter                    :: int                          :: maximum # of times to let the step size  :: ---
                                                                      optimization recurse before giving up
        uncs                       :: (Nkpar_surv,Nkperp_surv) of  :: unc in power spec @each cyl survey mode  :: K^2 Mpc^3 (same as power)
                                      floats
        frac_unc                   :: float                        :: if init w/ uncs=None, set uncs as        :: ---
                                                                      frac_unc*(fiducial power @survey modes) 
        Nkpar_box,Nkperp_box       :: ints                         :: # modes to put along cyl axes in power   :: ---
                                                                      spec calcs from boxes
        frac_tol_conv              :: float                        :: how much the Poisson noise must fall off :: ---
        pars_forecast_names        :: (N_pars_forecast,) or equiv. :: names of the pars being forecast         :: ---
                                      of strs
        manual_primary_beam_modes  :: x,y,z coordinate axes        :: domain of a discrete sampling            :: Mpc
                                      (if primary_beam !callable)
        per_ant_out_name           :: str                          :: name of per-antenna beam box to read     :: ---
                                                                      from or write to
        """
        # primary beam considerations
        if (primary_beam_categ.lower()=="uaa"):
            if (primary_beam_type.lower()=="gaussian" or primary_beam_type.lower()=="airy"):
                self.fwhm_x,self.fwhm_y=primary_beam_aux
                self.primary_beam_uncs= primary_beam_uncs
                self.epsx,self.epsy=    self.primary_beam_uncs
            else:
                raise NotYetImplementedError
        elif (primary_beam_categ.lower()=="pa" or primary_beam_categ.lower()=="manual"):
            if (primary_beam_categ.lower()=="pa"):
                # what happens in per_antenna? do I need to pass widths, or are those just calculated internally?
                # yeah, I guess for now it's just that I'm falling back on the circular cross-section lambda/D thing

                # self.box, self.xy_vec, self.z_vec
                fidu=per_antenna(nu_ctr=self.nu_ctr, N_pert_types=0) # fidu=CHORD_image(nu_ctr=test_freq, N_pert_types=0)
                fidu.stack_to_box(delta_nu=self.Deltanu,N_grid_pix=Npix)
                fidu_box=fidu.box
                pert=per_antenna(nu_ctr=self.nu_ctr, N_pert_types= , )

                # hmmmmm if the actual windowing calculation happens in a subsequent internal cosmo_stats call...
                # should I do it here to establish the sampled beam...
                # and then defer to manual mode
            
            # now do the manual-y things
            if (manual_primary_beam_modes is None):
                raise NotEnoughInfoError
            else:
                self.manual_primary_beam_modes=manual_primary_beam_modes
            try:
                self.manual_primary_fid,self.manual_primary_mis=primary_beam_aux # assumed to be sampled at the same config space points
            except: # primary beam samplings not unpackable the way they need to be
                raise NotEnoughInfoError
        else:
            raise PathologicalError # as far as primary power beam perturbations go, they can all pretty much be described as being applied UAA, PA, or in some externally-implemented custom way
        # elif (primary_beam_categ.lower()=="pa"):
        #     ///
        #     # what happens in per_antenna? do I need to pass widths, or are those just calculated internally?
        #     # yeah, I guess for now it's just that I'm falling back on the circular cross-section lambda/D thing

        #     # self.box, self.xy_vec, self.z_vec

        #     # hmmmmm if the actual windowing calculation happens in a subsequent internal cosmo_stats call...
        #     # should I do it here to establish the sampled beam...
        #     # and then defer to manual mode
        # elif (primary_beam_categ.lower()=="manual"):
        #     if (manual_primary_beam_modes is None):
        #         raise NotEnoughInfoError
        #     else:
        #         self.manual_primary_beam_modes=manual_primary_beam_modes
        #     try:
        #         self.manual_primary_fid,self.manual_primary_mis=primary_beam_aux # assumed to be sampled at the same config space points
        #     except: # primary beam samplings not unpackable the way they need to be
        #         raise NotEnoughInfoError
        # else:
        #     raise PathologicalError # as far as primary power beam perturbations go, they can all pretty much be described as being applied UAA, PA, or in some externally-implemented custom way

        self.primary_beam_type=primary_beam_type
        self.primary_beam_aux=primary_beam_aux
        self.primary_beam_uncs=primary_beam_uncs
        
        # forecasting considerations
        self.pars_set_cosmo=pars_set_cosmo
        self.N_pars_set_cosmo=len(pars_set_cosmo)
        self.pars_forecast=pars_forecast
        self.N_pars_forecast=len(pars_forecast)
        self.n_sph_modes=n_sph_modes
        self.dpar=dpar
        self.nu_ctr=nu_ctr
        self.Deltanu=delta_nu
        self.bw=nu_ctr*evol_restriction_threshold
        self.Nchan=int(self.bw/self.Deltanu)
        self.z_ctr=freq2z(nu_HI_z0,nu_ctr)
        self.nu_lo=self.nu_ctr-self.bw/2.
        self.z_hi=freq2z(nu_HI_z0,self.nu_lo)
        self.Dc_hi=comoving_distance(self.z_hi)
        self.nu_hi=self.nu_ctr+self.bw/2.
        self.z_lo=freq2z(nu_HI_z0,self.nu_hi)
        self.Dc_lo=comoving_distance(self.z_lo)
        self.deltaz=self.z_hi-self.z_lo
        self.surv_channels=np.arange(self.nu_lo,self.nu_hi,self.Deltanu)
        self.r0=comoving_distance(self.z_ctr)
        if (primary_beam_type.lower()=="gaussian" or primary_beam_type.lower()=="airy"):
            self.perturbed_primary_beam_aux=(self.fwhm_x*(1-self.epsx),self.fwhm_y*(1-self.epsy))
            self.primary_beam_aux=np.array([self.fwhm_x,self.fwhm_y,self.r0]) # UPDATING ARGS NOW THAT THE FULL SET HAS BEEN SPECIFIED
            self.perturbed_primary_beam_aux=np.append(self.perturbed_primary_beam_aux,self.r0)
        elif (primary_beam_type.lower()=="manual"):
            pass
        else:
            raise NotYetImplementedError

        # cylindrically binned survey k-modes and box considerations
        kpar_surv=kpar(self.nu_ctr,self.Deltanu,self.Nchan)
        self.ceil=ceil
        self.kpar_surv=kpar_surv
        if self.ceil>0:
            self.kpar_surv=self.kpar_surv[:-self.ceil]
        self.Nkpar_surv=len(self.kpar_surv)
        self.bmin=bmin
        self.bmax=bmax
        self.kperp_surv=kperp(self.nu_ctr,self.Nchan,self.bmin,self.bmax)
        self.Nkperp_surv=len(self.kperp_surv)

        # self.kmin_surv=np.min((self.kpar_surv[0 ],self.kperp_surv[0 ])) # no extrap issues but slow (mult. factors of 1.1 and 1.2 also incur this issue, to their respective extents)
        # self.kmin_surv=1.25*np.min((self.kpar_surv[ 0],self.kperp_surv[ 0])) # slight extrap issues and not so slow
        self.kmin_surv=np.min((self.kpar_surv[ 0],self.kperp_surv[ 0]))
        self.kmax_surv=np.sqrt(self.kpar_surv[-1]**2+self.kperp_surv[-1]**2)
        self.Lsurvbox= twopi/self.kmin_surv
        self.Nvoxbox=  int(self.Lsurvbox*self.kmax_surv/pi)
        print("Nvoxbox=",self.Nvoxbox)
        self.NvoxPracticalityWarning()

        # numerical protections for assorted k-ranges
        kmin_box_and_init=(1-init_and_box_tol)*self.kmin_surv
        kmax_box_and_init=(1+init_and_box_tol)*self.kmax_surv
        kmin_CAMB=(1-CAMB_tol)*kmin_box_and_init
        kmax_CAMB=(1+CAMB_tol)*kmax_box_and_init*np.sqrt(3) # factor of sqrt(3) from pythag theorem for box to prevent the need for extrap
        self.ksph,self.Ptruesph=self.get_mps(self.pars_set_cosmo,kmin_CAMB,kmax_CAMB)
        self.Deltabox=self.Lsurvbox/self.Nvoxbox
        if (primary_beam_type.lower()=="gaussian" or primary_beam_type.lower()=="airy"):
            sky_plane_sigmas=self.r0*np.array([self.fwhm_x,self.fwhm_y])/np.sqrt(2*np.log(2))
            self.all_sigmas=sky_plane_sigmas
            if (np.any(self.all_sigmas<self.Deltabox)):
                raise NumericalDeltaError
        elif (primary_beam_type.lower()=="manual"):
            print("WARNING: unable to do a robust numerical delta error check when a manual beam is passed")
        else:
            raise NotYetImplementedError

        # considerations for power spectra binned to survey k-modes
        _,_,self.Pcyl=self.unbin_to_Pcyl(self.pars_set_cosmo)
        self.frac_unc=frac_unc
        if (uncs==None):
            uncs=self.frac_unc*self.Pcyl
            uncs[uncs==0]=huge
            self.uncs=uncs
        else:
            self.uncs=uncs

        # precision control for numerical derivatives
        self.ftol_deriv=ftol_deriv
        self.eps=eps
        self.maxiter=maxiter

        # considerations for power spectrum binning directly from the box
        self.Nkpar_box=Nkpar_box
        self.Nkperp_box=Nkperp_box
        self.frac_tol_conv=frac_tol_conv
        self.no_monopole=no_monopole
        
        # considerations for printing the calculated bias results
        self.pars_forecast_names=pars_forecast_names
        assert (len(pars_forecast)==len(pars_forecast_names))

        # holder for numerical derivatives of a cylindrically binned power spectrum (sampled at the survey modes) wrt the params being forecast
        self.cyl_partials=np.zeros((self.N_pars_forecast,self.Nkpar_surv,self.Nkperp_surv))

    def get_mps(self,pars_use,minkh=1e-4,maxkh=1):
        """
        get matter power spectrum from CAMB
        """
        z=[self.z_ctr]
        H0=pars_use[0]
        h=H0/100.
        ombh2=pars_use[1]
        omch2=pars_use[2]
        As=pars_use[3]*scale
        ns=pars_use[4]

        # REAL PHYSICS
        pars_use_internal=camb.set_params(H0=H0, ombh2=ombh2, omch2=omch2, ns=ns, mnu=0.06,omk=0)
        pars_use_internal.InitPower.set_params(As=As,ns=ns,r=0)
        pars_use_internal.set_matter_power(redshifts=z, kmax=maxkh*h)
        results = camb.get_results(pars_use_internal)
        pars_use_internal.NonLinear = model.NonLinear_none
        kh,z,pk=results.get_matter_power_spectrum(minkh=minkh,maxkh=maxkh,npoints=self.n_sph_modes)

        # # NOT REAL PHYSICS (trying to tease out the k-dependence thing)
        # pk=-4*(kh-0.75)**2+2.5 # peak near 0.75, pos bw about 0 and 1.5 ... fine, but I think I'd be more targeted if I were to switch to a Gaussian "..._parabola"
        # pk=np.exp(-(kh-0.75)**2/(2*0.15**2))+0.25 # "..._gaussian"
        # pk=np.reshape(pk,(len(pk),1)) # although this seems like jumping through a convoluted hoop, the whole rest of the pipeline is referenced to the kind of (1,npts) shape you get from CAMB... easier to address in one line here rather than comment out all the other places, even if it's conceptually redundant
        # pk=pk.T
        return kh,pk
    
    def unbin_to_Pcyl(self,pars_to_use):
        """
        interpolate a spherically binned CAMB MPS to provide MPS values for a cylindrically binned k-grid of interest (nkpar x nkperp)
        """
        k,Psph_use=self.get_mps(pars_to_use,minkh=self.kmin_surv,maxkh=self.kmax_surv)
        Psph_use=Psph_use.reshape((Psph_use.shape[1],))
        self.Psph=Psph_use
        kpar_grid,kperp_grid=np.meshgrid(self.kpar_surv,self.kperp_surv,indexing="ij")
        kmag_grid=np.sqrt(kpar_grid**2+kperp_grid**2)

        kmag_grid_flat=np.reshape(kmag_grid,(self.Nkpar_surv*self.Nkperp_surv,))
        Psph_interpolator=interp1d(k,Psph_use,kind="cubic",bounds_error=False,fill_value="extrapolate")
        P_interp_flat=Psph_interpolator(kmag_grid_flat)
        Pcyl=np.reshape(P_interp_flat,(self.Nkpar_surv,self.Nkperp_surv))
        return kpar_grid,kperp_grid,Pcyl
    
    def NvoxPracticalityWarning(self,threshold_lo=75,threshold_hi=200):
        prefix="WARNING: the specified survey requires Nvox="
        if self.Nvoxbox>threshold_hi:
            print(prefix+"{:4}, which may cause slow eval".format(self.Nvoxbox))
        elif self.Nvoxbox<threshold_lo:
            print(prefix+"{:4}, which is suspiciously coarse".format(self.Nvoxbox))
        return None

    def calc_Pcont_asym(self):
        """
        calculate a cylindrically binned Pcont from an average over the power spectra formed from cylindrically-asymmetric-response-modulated brightness temp fields for a cosmological case of interest
        (you can still form a cylindrical summary statistic from brightness temp fields encoding effects beyond this symmetry)

        contaminant power, calculated as the difference of subtracted spectra with config space–multiplied "true" and "thought" instrument responses
        """
        if (self.primary_beam_type!="manual"):
            if (self.primary_beam_type=="Gaussian"):
                pb_here=UAA_Gaussian
            elif (self.primary_beam_type=="Airy"):
                pb_here=UAA_Airy
            else:
                raise NotYetImplementedError
            tr=cosmo_stats(self.Lsurvbox,
                           P_fid=self.Ptruesph,Nvox=self.Nvoxbox,
                           primary_beam=pb_here,primary_beam_aux=self.primary_beam_aux,
                           Nk0=self.Nkpar_box,Nk1=self.Nkperp_box,
                           frac_tol=self.frac_tol_conv,
                           k0bins_interp=self.kpar_surv,k1bins_interp=self.kperp_surv,
                           k_fid=self.ksph, no_monopole=self.no_monopole)
            th=cosmo_stats(self.Lsurvbox,
                           P_fid=self.Ptruesph,Nvox=self.Nvoxbox,
                           primary_beam=pb_here,primary_beam_aux=self.perturbed_primary_beam_aux,
                           Nk0=self.Nkpar_box,Nk1=self.Nkperp_box,
                           frac_tol=self.frac_tol_conv,
                           k0bins_interp=self.kpar_surv,k1bins_interp=self.kperp_surv,
                           k_fid=self.ksph, no_monopole=self.no_monopole)
        else:
            tr=cosmo_stats(self.Lsurvbox,
                           P_fid=self.Ptruesph,Nvox=self.Nvoxbox,
                           primary_beam=self.manual_primary_fid,primary_beam_type="manual",
                           Nk0=self.Nkpar_box,Nk1=self.Nkperp_box,
                           frac_tol=self.frac_tol_conv,
                           k0bins_interp=self.kpar_surv,k1bins_interp=self.kperp_surv,
                           k_fid=self.ksph,
                           manual_primary_beam_modes=self.manual_primary_beam_modes, no_monopole=self.no_monopole)
            th=cosmo_stats(self.Lsurvbox,
                           P_fid=self.Ptruesph,Nvox=self.Nvoxbox,
                           primary_beam=self.manual_primary_mis,primary_beam_type="manual",
                           Nk0=self.Nkpar_box,Nk1=self.Nkperp_box,
                           frac_tol=self.frac_tol_conv,
                           k0bins_interp=self.kpar_surv,k1bins_interp=self.kperp_surv,
                           k_fid=self.ksph,
                           manual_primary_beam_modes=self.manual_primary_beam_modes, no_monopole=self.no_monopole)
        tr.avg_realizations()
        th.avg_realizations()

        self.Ptrue_cyl=    tr.P_converged
        self.Pthought_cyl= th.P_converged
        self.Pcont_cyl=    self.Ptrue_cyl-self.Pthought_cyl ### same update as calc_Pcont_sym

        if (not np.all(self.Pcont_cyl.shape==self.uncs.shape)):
            interp_holder=cosmo_stats(self.Lsurvbox,P_fid=self.Pcont_cyl,Nvox=self.Nvoxbox,
                                      Nk0=self.Nkpar_box,Nk1=self.Nkperp_box,                                       
                                      k0bins_interp=self.kpar_surv,k1bins_interp=self.kperp_surv, 
                                      no_monopole=self.no_monopole) # hacky use of interpolate_P means the Nk0- and Nk1-determined bins will be treated as fiducial (or, at least, that's what I need to make happen)
            interp_holder.interpolate_P(use_P_fid=True)
            self.Pcont_cyl_surv=interp_holder.P_interp

            interp_holder=cosmo_stats(self.Lsurvbox,P_fid=self.Pthought_cyl,Nvox=self.Nvoxbox,
                                      Nk0=self.Nkpar_box,Nk1=self.Nkperp_box,                                       
                                      k0bins_interp=self.kpar_surv,k1bins_interp=self.kperp_surv, 
                                      no_monopole=self.no_monopole) # hacky use of interpolate_P means the Nk0- and Nk1-determined bins will be treated as fiducial (or, at least, that's what I need to make happen)
            interp_holder.interpolate_P(use_P_fid=True)
            self.Pthought_cyl_surv=interp_holder.P_interp

            interp_holder=cosmo_stats(self.Lsurvbox,P_fid=self.Ptrue_cyl,Nvox=self.Nvoxbox,
                                      Nk0=self.Nkpar_box,Nk1=self.Nkperp_box,                                       
                                      k0bins_interp=self.kpar_surv,k1bins_interp=self.kperp_surv, 
                                      no_monopole=self.no_monopole) # hacky use of interpolate_P means the Nk0- and Nk1-determined bins will be treated as fiducial (or, at least, that's what I need to make happen)
            interp_holder.interpolate_P(use_P_fid=True)
            self.Ptrue_cyl_surv=interp_holder.P_interp
            print("interpolated Pcont, Ptrue, and Pthought to survey modes")
        else: # no interpolation necessary
            self.Pcont_cyl_surv=self.Pcont_cyl
            self.Pthought_cyl_surv=self.Pthought_cyl
            self.Ptrue_cyl_surv=self.Ptrue_cyl

    def cyl_partial(self,n):  
        """        
        cylindrically binned matter power spectrum partial WRT one cosmo parameter (nkpar x nkperp)
        """
        dparn=self.dpar[n]
        pcopy=self.pars_set_cosmo.copy()
        pndispersed=pcopy[n]+np.linspace(-2,2,5)*dparn

        P0=np.mean(np.abs(self.Pcyl))+self.eps
        tol=self.ftol_deriv*P0 # generalizes tol=ftol*f0 from 512

        pcopy[n]=pcopy[n]+2*dparn # don't need to generate a fresh copy immediately before b/c the initial copy hasn't been modified yet
        _,_,Pcyl_2plus=self.unbin_to_Pcyl(pcopy)
        pcopy=self.pars_set_cosmo.copy()
        pcopy[n]=pcopy[n]-2*dparn
        _,_,Pcyl_2minu=self.unbin_to_Pcyl(pcopy)
        deriv1=(Pcyl_2plus-Pcyl_2minu)/(4*self.dpar[n])

        pcopy=self.pars_set_cosmo.copy()
        pcopy[n]=pcopy[n]+dparn
        _,_,Pcyl_plus=self.unbin_to_Pcyl(pcopy)
        pcopy=self.pars_set_cosmo.copy()
        pcopy[n]=pcopy[n]-dparn
        _,_,Pcyl_minu=self.unbin_to_Pcyl(pcopy)
        deriv2=(Pcyl_plus-Pcyl_minu)/(2*self.dpar[n])

        Pcyl_dif=Pcyl_plus-Pcyl_minu
        if (np.mean(Pcyl_dif)<tol): # consider relaxing this to np.any if it ever seems like too strict a condition?!
            estimate=(4*deriv2-deriv1)/3
            self.iter=0 # reset for next time
            self.cyl_partials[n,:,:]=estimate
        else:
            pnmean=np.mean(np.abs(pndispersed)) # the np.abs part should be redundant because, by this point, all the k-mode values and their corresponding dpns and Ps should be nonnegative, but anyway... numerical stability or something idk
            Psecond=np.abs(np.mean(2*self.Pcyl-Pcyl_minu-Pcyl_plus))/self.dpar[n]**2
            dparn=np.sqrt(self.eps*pnmean*P0/Psecond)
            self.dpar[n]=dparn # send along knowledge of the updated step size
            self.iter+=1
            self.cyl_partial(n) # recurse
            if self.iter==self.maxiter:
                print("failed to converge in {:d} iterations".format(self.maxiter))
                fallback=(4*deriv2-deriv1)/3
                print("RETURNING fallback")
                self.iter=0 # still need to reset for next time
                self.cyl_partials[n,:,:]=fallback

    def build_cyl_partials(self):
        """
        builds a (N_pars_forecast,Nkpar,Nkperp) array of the partials of the cylindrically binned MPS WRT each cosmo param in the forecast
        """
        for n in range(self.N_pars_set_cosmo):
            self.iter=0 # bc starting a new partial deriv calc.
            self.cyl_partial(n)
        
    def bias(self):
        """
        collect and stitch together the ingredients of the parameter bias calculation
        """
        self.build_cyl_partials()
        print("built partials")
        self.calc_Pcont_asym()
        print("computed Pcont")
        V=0.*self.cyl_partials
        for i in range(self.N_pars_forecast):
            V[i,:,:]=self.cyl_partials[i,:,:]/self.uncs # elementwise division for an nkpar x nkperp slice
        V_completely_transposed=np.transpose(V,axes=(2,1,0))
        F=np.einsum("ijk,kjl->il",V,V_completely_transposed)
        print("computed F")
        Pcont_div_sigma=self.Pcont_cyl_surv/self.uncs
        B=np.einsum("jk,ijk->i",Pcont_div_sigma,V)
        print("computed B")
        self.biases=(np.linalg.inv(F)@B).reshape((self.N_pars_forecast,))
        print("computed b")

    def print_survey_characteristics(self):
        print("survey properties.......................................................................")
        print("........................................................................................")
        print("survey centred at.......................................................................\n    nu ={:>7.4}     MHz \n    z  = {:>9.4} \n    Dc = {:>9.4f}  Mpc\n".format(self.nu_ctr,self.z_ctr,self.r0))
        print("survey spans............................................................................\n    nu =  {:>5.4}    -  {:>5.4}    MHz (deltanu = {:>6.4}    MHz) \n    z =  {:>9.4} - {:>9.4}     (deltaz  = {:>9.4}    ) \n    Dc = {:>9.4f} - {:>9.4f} Mpc (deltaDc = {:>9.4f} Mpc)\n".format(self.nu_lo,self.nu_hi,self.bw,self.z_hi,self.z_lo,self.z_hi-self.z_lo,self.Dc_hi,self.Dc_lo,self.Dc_hi-self.Dc_lo))
        if (self.primary_beam_type.lower()!="manual"):
            print("characteristic instrument response widths...............................................\n    beamFWHM0 = {:>8.4}  rad (frac. uncert. {:>7.4})\n".format(self.fwhm_x,self.epsx))
            print("specific to the cylindrically asymmetric beam...........................................\n    beamFWHM1 = {:>8.4}  rad (frac. uncert. {:>7.4})\n".format(self.fwhm_y,self.epsy))
        print("cylindrically binned wavenumbers of the survey..........................................\n    kparallel {:>8.4} - {:>8.4} Mpc**(-1) ({:>4} channels of width {:>7.4}  Mpc**(-1)) \n    kperp     {:>8.4} - {:>8.4} Mpc**(-1) ({:>4} bins of width {:>8.4} Mpc**(-1))\n".format(self.kpar_surv[0],self.kpar_surv[-1],self.Nkpar_surv,self.kpar_surv[-1]-self.kpar_surv[-2],   self.kperp_surv[0],self.kperp_surv[-1],self.Nkperp_surv,self.kperp_surv[-1]-self.kperp_surv[-2]))
        print("cylindrically binned k-bin sensitivity..................................................\n    fraction of Pcyl amplitude = {:>7.4}".format(self.frac_unc))

    def print_results(self):
        print("\n\nbias calculation results for the survey described above.................................")
        print("........................................................................................")
        for p,par in enumerate(self.pars_forecast):
            print('{:12} = {:-10.3e} with bias {:-12.5e} (fraction = {:-10.3e})'.format(self.pars_forecast_names[p], par, self.biases[p], self.biases[p]/par))
        return None

"""
this class helps connect ensemble-averaged power spectrum estimates and 
cosmological brighness temperature boxes for assorted interconnected use cases:
1. generate a power spectrum that describes the statistics of a cosmo box
2. generate realizations of a cosmo box consistent with a known power spectrum
3. iterate power spec calcs from different box realizations until convergence
4. interpolate a power spectrum (sph, cyl, or sph->grid)
"""

class cosmo_stats(object):
    def __init__(self,
                 Lxy,Lz=None,                                                            # one scaling is nonnegotiable for box->spec and spec->box calcs; the other would be useful for rectangular prism box considerations (sky plane slice is square, but LoS extent can differ)
                 T_pristine=None,T_primary=None,P_fid=None,Nvox=None,Nvoxz=None,         # need one of either T (pristine or primary) or P to get started; I also check for any conflicts with Nvox
                 primary_beam=None,primary_beam_aux=None,primary_beam_type="Gaussian",  # primary beam considerations
                 Nk0=10,Nk1=0,binning_mode="lin",                                        # binning considerations for power spec realizations (log mode not fully tested yet b/c not impt. for current pipeline)
                 frac_tol=0.1,                                                           # max number of realizations
                 k0bins_interp=None,k1bins_interp=None,                                  # bins where it would be nice to know about P_converged
                 P_realizations=None,P_converged=None,                                   # power spectra related to averaging over those from dif box realizations
                 verbose=False,                                                          # status updates for averaging over realizations
                 k_fid=None,kind="cubic",avoid_extrapolation=False,                      # helper vars for converting a 1d fid power spec to a box sampling
                 no_monopole=True,                                                       # consideration when generating boxes
                 manual_primary_beam_modes=None,                                         # when using a discretely sampled primary beam not sampled internally using a callable, it is necessary to provide knowledge of the modes at which it was sampled
                 ):                                                                      # implement soon: synthesized beam considerations, other primary beam types, and more
        """
        Lxy,Lz                    :: float                       :: side length of cosmo box          :: Mpc
        T_pristine                :: (Nvox,Nvox,Nvox) of floats  :: cosmo box (just physics/no beam)  :: K
        T_primary                 :: (Nvox,Nvox,Nvox) of floats  :: cosmo box * primary beam          :: K
        P_fid                     :: (Nk0_fid,) of floats        :: sph binned fiducial power spec    :: K^2 Mpc^3
        Nvox,Nvoxz                :: float                       :: cosmobox#vox/side,z-ax can differ :: ---
        primary_beam              :: callable (or, if            :: power beam in Cartesian coords    :: ---
                                     primary_beam_type=="manual" 
                                     a 3D array)          
        primary_beam_aux         :: tuple of floats             :: Gaussian, AiryGaussian: μ, σ      :: Gaussian: r0 in Mpc; fwhm_x, fwhm_y in rad
        primary_beam_type         :: str                         :: for now: Gaussian / AiryGaussian  :: ---
        Nk0, Nk1                  :: int                         :: # power spec bins for axis 0/1    :: ---
        binning_mode              :: str                         :: lin/log sp. P_realizations bins   :: ---
        frac_tol                  :: float                       :: max fractional amount by which    :: ---
                                                                    the p.s. avg can change w/ the 
                                                                    addition of the latest realiz. 
                                                                    and the ensemble average is 
                                                                    considered converged
        k0bins_interp,            :: (Nk0_interp,) of floats     :: bins to which to interpolate the  :: 1/Mpc
        k1bins_interp                (Nk1_interp,) of floats        converged power spec (prob set
                                                                    by survey considerations)
        P_realizations            :: if Nk1==0: (Nk0,)    floats :: sph/cyl power specs for dif       :: K^2 Mpc^3
                                     if Nk1>0:  (Nk0,Nk1) floats    realizations of a cosmo box 
        P_converged               :: same as that of P_fid       :: average of P_realizations         :: K^2 Mpc^3
        verbose                   :: bool                        :: every 10% of realization_ceil     :: ---
        k_fid                     :: (Nk0_fid,) of floats        :: modes where P_fid is sampled      :: 1/Mpc
        kind                      :: str                         :: interp type                       :: ---
        avoid_extrapolation       :: bool                        :: when calling scipy interpolators  :: ---
        no_monopole               :: bool                        :: y/n subtr. from generated boxes   :: ---
        manual_primary_beam_modes :: primary_beam.shape of       :: domain of a discrete sampling     :: Mpc
                                     floats (when not callable) 
        """
        # spectrum and box
        if (Lz is None): # cubic box
            self.Lz=Lxy
            self.Lxy=Lxy
        else:            # the kind of rectangular prism box I care about for dirty image stacking
            self.Lz=Lz
            self.Lxy=Lxy
        self.P_fid=P_fid
        self.T_primary=T_primary
        self.T_pristine=T_pristine
        self.no_monopole=no_monopole
        if ((T_primary is None) and (T_pristine is None) and (P_fid is None)): # require either a box or a fiducial power spec (il faut some way of determining #voxels/side; passing just Nvox is not good enough)
            raise NotEnoughInfoError
        else:                                                                  # there is possibly enough info to proceed, but still need to check for conflicts and gaps
            if ((T_pristine is not None) and (T_primary is not None)):
                print("WARNING: T_pristine and T_primary both passed; T_primary will be temporarily ignored and then internally overwritten to ensure consistency with primary_beam")
                if (T_pristine.shape!=T_primary.shape):
                    raise ConflictingInfoError
                else:                                                          # use box shape to set cubic/ rectangular prism box attributes
                    self.Nvox,_,self.Nvoxz=T_primary.shape
            if ((Nvox is not None) and (T_pristine is not None)):              # possible conflict: if both Nvox and a box are passed, 
                T_pristine_shape0,_,T_pristine_shape2=T_pristine.shape
                if (Nvox!=T_pristine.shape[0]):                                # but Nvox and the box shape disagree,
                    raise ConflictingInfoError                                 # estamos en problemas
                else:
                    self.Nvox= T_pristine_shape0                               # otherwise, initialize the Nvox attributes
                    self.Nvoxz=T_pristine_shape2
            elif (Nvox is not None):                                           # if Nvox was passed but T was not, use Nvox to initialize the Nvox attributes
                self.Nvox=Nvox 
                if (Nvoxz is None):                                            # if no Nvoxz was provided, make the box cubic
                    self.Nvoxz=Nvox
                else:
                    self.Nvoxz=Nvoxz
            else:                                                              # remaining case: T was passed but Nvox was not, so use the shape of T to initialize the Nvox attributes
                self.Nvox= T_pristine_shape0
                self.Nvoxz=T_pristine_shape2
            if ((T_primary is not None) and (T_pristine is None)):             # passing T_primary but not T_pristine is not handled anywhere up to this point
                self.Nvox,_,self.Nvoxz=T_primary.shape
            
            if (P_fid is not None): # no hi fa res si the fiducial power spectrum has a different dimensionality or bin width than the realizations you plan to generate (boxes will be generated from a grid-interpolated P_fid, anyway)
                Pfidshape=P_fid.shape
                Pfiddims=len(Pfidshape)
                if (Pfiddims==2):
                    if primary_beam is None: # trying to do a minimalistic instantiation where I merely provide a fiducial power spectrum and interpolate it
                        self.fid_Nk0,self.fid_Nk1=Pfidshape
                    else:
                        try: # see if the power spec is a CAMB-esque (1,npts) array
                            self.P_fid=np.reshape(P_fid,(Pfidshape[-1],)) # make the CAMB MPS shape amenable to the calcs internal to this class
                        except: # barring that...
                            pass # treat the power spectrum as being truly cylindrically binned
                elif (Pfiddims==1):
                    self.fid_Nk0=Pfidshape[0] # already checked that P_fid is 1d, so no info is lost by extracting the int in this one-element tuple, and fid_Nk0 being an integer makes things work the way they should down the line
                    self.fid_Nk1=0
                else:
                    raise UnsupportedBinningMode
        
        # config space
        self.Deltaxy=self.Lxy/self.Nvox                           # sky plane: voxel side length
        self.xy_vec_for_box=self.Lxy*fftshift(fftfreq(self.Nvox)) # sky plane Cartesian config space coordinate axis
        self.Deltaz= self.Lz/self.Nvoxz                           # line of sight voxel side length
        self.z_vec_for_box= self.Lz*fftshift(fftfreq(self.Nvoxz)) # line of sight Cartesian config space coordinate axis
        self.d3r=self.Deltaz*self.Deltaxy**2                      # volume element = voxel volume

        self.xx_grid,self.yy_grid,self.zz_grid=np.meshgrid(self.xy_vec_for_box,
                                                           self.xy_vec_for_box,
                                                           self.z_vec_for_box,
                                                           indexing="ij")      # box-shaped Cartesian coords
        self.r_grid=np.sqrt(self.xx_grid**2+self.yy_grid**2+self.zz_grid**2)   # r magnitudes at each voxel

        # Fourier space
        self.Deltakxy=twopi/self.Lxy                                        # voxel side length
        self.Deltakz= twopi/self.Lz
        self.d3k=self.Deltakxy**2*self.Deltakz                              # volume element / voxel volume
        self.kxy_vec_for_box_corner=twopi*fftfreq(self.Nvox,d=self.Deltaxy) # one Cartesian coordinate axis - non-fftshifted/ corner origin
        self.kz_vec_for_box_corner= twopi*fftfreq(self.Nvoxz,d=self.Deltaz)
        self.kx_grid_corner,self.ky_grid_corner,self.kz_grid_corner=np.meshgrid(self.kxy_vec_for_box_corner,
                                                                                self.kxy_vec_for_box_corner,
                                                                                self.kz_vec_for_box_corner,
                                                                                indexing="ij")               # box-shaped Cartesian coords
        self.kmag_grid_corner= np.sqrt(self.kx_grid_corner**2+self.ky_grid_corner**2+self.kz_grid_corner**2) # k magnitudes for each voxel (need for the generate_box direction)
        self.kmag_grid_centre=fftshift(self.kmag_grid_corner) 
        self.kmag_grid_corner_flat=np.reshape(self.kmag_grid_corner,(self.Nvox**2*self.Nvoxz,))

        # if P_fid was passed, establish its values on the k grid (helpful when generating a box)
        self.k_fid=k_fid
        self.kind=kind
        self.avoid_extrapolation=avoid_extrapolation
        if (self.P_fid is not None and self.k_fid is not None):
            if (len(self.P_fid.shape)==1): # truly 1d fiducial power spec (by this point, even CAMB-like shapes have been reshuffled)
                self.P_fid_interp_1d_to_3d()
            elif (len(self.P_fid.shape)==2):
                self.k_fid0,self.kfid1=self.k_fid # fiducial k-modes should be unpackable, since P_fid has been verified to be truly 2d
                self.P_fid_interp_2d_to_3d()
            else: # so far, I do not anticipate working with "truly three dimensional"/ unbinned power spectra
                raise NotYetImplementedError
        
        # binning considerations
        self.binning_mode=binning_mode
        self.Nk0=Nk0 # the number of bins to put in power spec realizations you construct (ok if not the same as the number of bins in the fiducial power spec)
        self.Nk1=Nk1
        self.kmax_box_xy= pi/self.Deltaxy
        self.kmax_box_z=  pi/self.Deltaz
        self.kmin_box_xy= twopi/self.Lxy
        self.kmin_box_z=  twopi/self.Lz
        self.k0bins,self.limiting_spacing_0=self.calc_bins(self.Nk0,self.Nvoxz,self.kmin_box_z,self.kmax_box_z)
        if self.limiting_spacing_0<self.Deltakz: # trying to bin more finely than the box can tell you about (guaranteed to have >=1 empty bin)
            raise ResolutionError
        
        if (self.Nk1>0):
            self.k1bins,self.limiting_spacing_1=self.calc_bins(self.Nk1,self.Nvox,self.kmin_box_xy,self.kmax_box_xy)
            if (self.limiting_spacing_1<self.Deltakxy): # idem ^
                raise ResolutionError
        else:
            self.k1bins=None
        
            # voxel grids for sph binning        
        self.sph_bin_indices_centre=      np.digitize(self.kmag_grid_centre,self.k0bins)
        self.sph_bin_indices_1d_centre=   np.reshape(self.sph_bin_indices_centre, (self.Nvox**2*self.Nvoxz,))

            # voxel grids for cyl binning
        if (self.Nk1>0):
            self.kpar_column_centre= np.abs(fftshift(self.kz_vec_for_box_corner))                                      # magnitudes of kpar for a representative column along the line of sight (z-like)
            self.kperp_slice_centre= np.sqrt(fftshift(self.kx_grid_corner)**2+fftshift(self.ky_grid_corner)**2)[:,:,0] # magnitudes of kperp for a representative slice transverse to the line of sight (x- and y-like)
            self.perpbin_indices_slice_centre=    np.digitize(self.kperp_slice_centre,self.k1bins)                     # cyl kperp bin that each voxel falls into
            self.perpbin_indices_slice_1d_centre= np.reshape(self.perpbin_indices_slice_centre,(self.Nvox**2,))        # 1d version of ^ (compatible with np.bincount)
            self.parbin_indices_column_centre=    np.digitize(self.kpar_column_centre,self.k0bins)                     # cyl kpar bin that each voxel falls into

        # primary beam
        self.primary_beam=primary_beam
        self.primary_beam_aux=primary_beam_aux
        self.primary_beam_type=primary_beam_type
        self.manual_primary_beam_modes=manual_primary_beam_modes
        if (self.primary_beam is not None): # non-identity primary beam
            if (self.primary_beam_type=="Gaussian" or self.primary_beam_type=="AiryGaussian"):
                self.fwhm_x,self.fwhm_y,self.r0=self.primary_beam_aux
                evaled_primary=self.primary_beam(self.xx_grid,self.yy_grid,self.fwhm_x,self.fwhm_y,self.r0)
            elif (self.primary_beam_type=="manual"):
                try:    # to access this branch, the manual/ numerically sampled primary beam needs to be close enough to a numpy array that it has a shape and not, e.g. a callable
                    primary_beam.shape
                except: # primary beam is a callable (or something else without a shape method), which is not in line with how this part of the code is supposed to work
                    raise ConflictingInfoError 
                if self.manual_primary_beam_modes is None:
                    raise NotEnoughInfoError

                print("start... of manual primary beam extrapolation warning checks")
                x_manual_primary,y_manual_primary,z_manual_primary=manual_primary_beam_modes
                x_have_lo=x_manual_primary[0]
                x_have_hi=x_manual_primary[-1]
                y_have_lo=y_manual_primary[0]
                y_have_hi=y_manual_primary[-1]
                z_have_lo=z_manual_primary[0]
                z_have_hi=z_manual_primary[-1]
                xy_want_lo=self.xy_vec_for_box[0]
                xy_want_hi=self.xy_vec_for_box[-1]
                z_want_lo=self.z_vec_for_box[0]
                z_want_hi=self.z_vec_for_box[-1]
                if (xy_want_lo<x_have_lo):
                    extrapolation_warning("low x",   xy_want_lo,  x_have_lo)
                if (xy_want_hi>x_have_hi):
                    extrapolation_warning("high x",   xy_want_hi,  x_have_hi)
                if (xy_want_lo<y_have_lo):
                    extrapolation_warning("low y",   xy_want_lo,  y_have_lo)
                if (xy_want_hi>y_have_hi):
                    extrapolation_warning("high y",   xy_want_hi,  y_have_hi)
                if (z_want_lo<z_have_lo):
                    extrapolation_warning("low z",   z_want_lo,  z_have_lo)
                if (z_want_hi>z_have_hi):
                    extrapolation_warning("high z",   z_want_hi,  z_have_hi)
                print("end..... of manual primary beam extrapolation warning checks")
                evaled_primary=interpn(manual_primary_beam_modes,
                                       self.primary_beam,
                                       (self.xx_grid,self.yy_grid,self.zz_grid),
                                       method=self.kind,bounds_error=self.avoid_extrapolation,fill_value=None)
            else:
                raise NotYetImplementedError
            self.Veff=np.sum(evaled_primary*self.d3r)           # rectangular sum method

            fft_evaled_primary=fftshift(fftn(ifftshift(evaled_primary)*self.d3r))
            beamtildesq_values=(fft_evaled_primary*np.conj(fft_evaled_primary)).real
            beamtildesq=maxfloat*np.ones((self.Nvox,self.Nvox,self.Nvoxz))
            use=np.nonzero(beamtildesq_values!=0.)
            beamtildesq[use]=beamtildesq_values[use]
            self.beamtildesq=beamtildesq
            self.beamtildesq=1 # placeholder

            xidx=self.Nvox//2
            yidx=self.Nvox//2
            zidx=self.Nvoxz//2
            beam_x_slice=evaled_primary[:,yidx,zidx]
            beam_x_norm=np.sum(beam_x_slice**2)/self.Nvox # adds up to 1 if the evaled primary beam is 1 everywhere
            beam_y_slice=evaled_primary[xidx,:,zidx]
            beam_y_norm=np.sum(beam_y_slice**2)/self.Nvox
            beam_z_slice=evaled_primary[xidx,yidx,:]
            beam_z_norm=np.sum(beam_z_slice**2)/self.Nvoxz
            self.beam_norm=beam_x_norm*beam_y_norm*beam_z_norm

            evaled_primary_for_div=np.copy(evaled_primary)
            evaled_primary_for_mul=np.copy(evaled_primary)
            evaled_primary_for_div[evaled_primary_for_div<nearly_zero]=maxfloat # protect against division-by-zero errors
            self.evaled_primary_for_div=evaled_primary_for_div
            self.evaled_primary_for_mul=evaled_primary_for_mul
        else:                               # identity primary beam
            self.Veff=self.Lxy**2*self.Lz
            self.evaled_primary_for_div=np.ones((self.Nvox,self.Nvox,self.Nvoxz))
            self.evaled_primary_for_mul=np.copy(self.evaled_primary_for_div)
            self.beamtildesq=np.ones((self.Nvox,self.Nvox,self.Nvoxz))
            self.beam_norm=1
        if (self.T_pristine is not None):
            self.T_primary=self.T_pristine*self.evaled_primary
        ############
        
        # strictness control for realization averaging
        self.frac_tol=frac_tol
        self.realization_ceiling=int(np.round(self.frac_tol**-2))
        self.verbose=verbose

        # P_converged interpolation bins
        self.k0bins_interp=k0bins_interp
        self.k1bins_interp=k1bins_interp

        # realization, averaging, and interpolation placeholders if no prior info
        if (P_realizations is not None):       # maybe you want to import realizations from a prev run and just add more? (unclear why you'd have left the
            self.P_realizations=P_realizations # prev run w/o a converged average, unless, maybe, you want to re-run with a stricter convergence threshold?)
        else:
            self.P_realizations=[] 
        if (P_converged is not None):          # maybe you have a converged power spec average from a previous calc and just want to interpolate or generate more box realizations?
            self.P_converged=P_converged
        else:
            self.P_converged=None
        self.P_interp=None                     # can't init with this because, if you had one, there'd be no point of using cosmo_stats b/c the job is already done (at best, you can provide a P_fid)
        self.not_converged=None

    def calc_bins(self,Nki,Nvox_to_use,kmin_to_use,kmax_to_use):
        """
        generate a set of bins spaced according to the desired scheme with max and min
        """
        if (self.binning_mode=="log"):
            kbins=np.logspace(np.log10(kmin_to_use),np.log10(kmax_to_use),num=Nki)
            limiting_spacing=twopi*(10.**(kmax_to_use)-10.**(kmax_to_use-(np.log10(Nvox_to_use)/Nki))) # twopi*(10.**(self.kmax_box)-10.**(self.kmax_box-(np.log10(self.Nvox)/Nki)))
        elif (self.binning_mode=="lin"):
            kbins=np.linspace(kmin_to_use,kmax_to_use,Nki)
            limiting_spacing=twopi*(0.5*Nvox_to_use-1)/(Nki) # version for a kmax that is "aware that" there are +/- k-coordinates in the box
        else:
            raise UnsupportedBinningMode
        return kbins,limiting_spacing # kbins            -> floors of the bins to which the power spectrum will be binned (along one axis)
                                      # limiting_spacing -> smallest spacing between adjacent bins (uniform if linear; otherwise, depends on the binning strategy)
    
    def P_fid_interp_1d_to_3d(self):
        """
        * interpolate a "physics-only" (spherically symmetric) power spectrum (e.g. from CAMB) to a 3D cosmological box.
        * for now, I don't have a solution better than overwriting the k=0 term after the fact (because extrapolation to this term based on a reasonable CAMB call leads to negative power there)
        """
        P_fid_interpolator=interp1d(self.k_fid,self.P_fid,kind=self.kind,bounds_error=self.avoid_extrapolation,fill_value="extrapolate")
        P_interp_flat=P_fid_interpolator(self.kmag_grid_corner_flat)
        self.P_fid_box=np.reshape(P_interp_flat,(self.Nvox,self.Nvox,self.Nvoxz))
            
    def generate_P(self,send_to_P_fid=False,T_use=None):
        """
        philosophy: 
        * compute the power spectrum of a known cosmological box and bin it spherically or cylindrically
        * append to the list of reconstructed P realizations (self.P_realizations)
        """
        if T_use is None:
            T_use=self.T_pristine
        else:
            T_use=self.T_primary
        if (self.T_pristine is None):    # power spec has to come from a box
            self.generate_box() # populates/overwrites self.T_pristine and self.T_primary
        T_tilde=            fftshift(fftn((ifftshift(T_use)*self.d3r)))
        modsq_T_tilde=     (T_tilde*np.conjugate(T_tilde)).real
        modsq_T_tilde[:,:,self.Nvoxz//2]*=2 # fix pos/neg duplication issue at the origin
        modsq_T_tilde/=self.beamtildesq
        if (self.Nk1==0):   # bin to sph
            modsq_T_tilde_1d= np.reshape(modsq_T_tilde,    (self.Nvox**2*self.Nvoxz,))

            sum_modsq_T_tilde= np.bincount(self.sph_bin_indices_1d_centre, 
                                           weights=modsq_T_tilde_1d, 
                                           minlength=self.Nk0)       # for the ensemble avg: sum    of modsq_T_tilde values in each bin
            N_modsq_T_tilde=   np.bincount(self.sph_bin_indices_1d_centre,
                                           minlength=self.Nk0)       # for the ensemble avg: number of modsq_T_tilde values in each bin
            sum_modsq_T_tilde_truncated=sum_modsq_T_tilde[:-1]       # excise sneaky corner modes: I devised my binning to only tell me about voxels w/ k<=(the largest sphere fully enclosed by the box), and my bin edges are floors. But, the highest floor corresponds to the point of intersection of the box and this largest sphere. To stick to my self-imposed "the stats are not good enough in the corners" philosophy, I must explicitly set aside the voxels that fall into the "catchall" uppermost bin. 
            N_modsq_T_tilde_truncated=  N_modsq_T_tilde[:-1]         # idem ^
            final_shape=(self.Nk0,)
        elif (self.Nk0!=0): # bin to cyl
            sum_modsq_T_tilde= np.zeros((self.Nk0+1,self.Nk1+1)) # for the ensemble avg: sum    of modsq_T_tilde values in each bin  ...upon each access, update the kparBIN row of interest, but all Nkperp columns
            N_modsq_T_tilde=   np.zeros((self.Nk0+1,self.Nk1+1)) # for the ensemble avg: number of modsq_T_tilde values in each bin
            for i in range(self.Nvoxz): # iterate over the kpar axis of the box to capture all LoS slices
                if (i==0): # stats for the representative "bull's eye" slice transverse to the LoS
                    slice_bin_counts=np.bincount(self.perpbin_indices_slice_1d_centre, minlength=self.Nk1)
                modsq_T_tilde_slice= modsq_T_tilde[:,:,i]                    # take the slice of interest of the preprocessed box values !!kpar is z-like
                modsq_T_tilde_slice_1d= np.reshape(modsq_T_tilde_slice, 
                                                   (self.Nvox**2,))          # 1d for bincount compatibility
                current_binsums= np.bincount(self.perpbin_indices_slice_1d_centre,
                                             weights=modsq_T_tilde_slice_1d, 
                                             minlength=self.Nk1)             # this slice's update to the numerator of the ensemble average
                current_par_bin=self.parbin_indices_column_centre[i]

                sum_modsq_T_tilde[current_par_bin,:]+= current_binsums  # update the numerator   of the ensemble avg
                N_modsq_T_tilde[  current_par_bin,:]+= slice_bin_counts # update the denominator of the ensemble avg
            
            sum_modsq_T_tilde_truncated= sum_modsq_T_tilde[:-1,:-1] # excise sneaky corner modes (see the analogous operation in the sph branch for an explanation)
            N_modsq_T_tilde_truncated=   N_modsq_T_tilde[  :-1,:-1] # idem ^
            final_shape=(self.Nk0,self.Nk1)

        N_modsq_T_tilde_truncated[N_modsq_T_tilde_truncated==0]=maxint # avoid division-by-zero errors during the division the estimator demands

        avg_modsq_T_tilde=sum_modsq_T_tilde_truncated/(N_modsq_T_tilde_truncated) # actual estimator math
        denom=self.Veff*self.beam_norm
        P=np.array(avg_modsq_T_tilde/denom)
        P.reshape(final_shape)
        if send_to_P_fid: # if generate_P was called speficially to have a spec from which all future box realizations will be generated
            self.P_fid=P
            self.P_fid_interp_1d_to_3d() # generate interpolated values of the newly established 1D P_fid over the k-magnitudes of the box
        else:             # the "normal" case where you're just accumulating a realization
            self.P_realizations.append([P])
        self.unbinned_P=modsq_T_tilde/denom # box-shaped, but calculated according to the power spectrum estimator equation
        
    def generate_box(self):
        """
        philosophy: 
        * generate a box that comprises a random realization of a known power spectrum
        * this always generates a "pristine" box and stores it in self.T_pristine
        * if primary_beam is not None, self.T_beamed is also populated
        """
        assert(self.Nvox>=self.Nk0), PathologicalError
        if (self.P_fid is None):
            try:
                self.generate_P(store_as_P_fid=True) # T->P_fid is deterministic, so, even if you start with a random realization, it'll be helpful to have a power spec summary stat to generate future realizations
            except: # something goes wrong in the P_fid calculation
                raise NotEnoughInfoError
        # not checking for 2D-ness of P_fid here since I've already done that during init (yes, support for cyl binned P_fid is still functionality I want to add eventually)
        # not warning abt potentially overwriting T -> the only case where info would be lost is where self.P_fid is None, and I already have a separate warning for that
        
        assert(self.P_fid_box is not None)
        if (self.Veff<=0):
            raise PathologicalError
        if (np.min(self.P_fid_box)<0):
            self.P_fid_box[self.P_fid_box<0]=0 # hackily overwriting error from having to extrapolate at the origin
        sigmas=np.sqrt(self.Veff*self.P_fid_box/2.) # from inverting the estimator equation and turning variances into std devs
        T_tilde_Re,T_tilde_Im=np.random.normal(loc=0.*sigmas,scale=sigmas,size=np.insert(sigmas.shape,0,2))
        
        T_tilde=T_tilde_Re+1j*T_tilde_Im # have not yet applied the symmetry that ensures T is real-valued 
        T=fftshift(irfftn(T_tilde*self.d3k,s=(self.Nvox,self.Nvox,self.Nvoxz),axes=(0,1,2),norm="forward"))/(twopi)**3 # handle in one line: fftshiftedness, ensuring T is real-valued and box-shaped, enforcing the cosmology Fourier convention
        if self.no_monopole:
            T-=np.mean(T) # subtract monopole moment to make things more akin to what powerbox does
        self.T_pristine=T
        self.T_primary=T*self.evaled_primary_for_mul

    def avg_realizations(self):
        """
        philosophy:
        * since P->box is not deterministic,
        * compute the power spectra from a bunch of generated boxes and average them together
        * realization ceiling precalculated from the Poisson noise–related fractional tolerance
        """
        assert(self.P_fid is not None), "cannot average over numerically windowed realizations without a fiducial power spec"
        self.not_converged=True
        i=0

        for i in range(self.realization_ceiling):
            self.generate_box()
            self.generate_P(T_use="primary")
            if self.verbose:
                if (i%(self.realization_ceiling//10)==0):
                    print("realization",i)

        arr_realiz_holder=np.array(self.P_realizations)
        if (arr_realiz_holder.shape[0]>1):
            P_converged=np.mean(arr_realiz_holder,axis=0)
        else:
            P_converged=arr_realiz_holder

        if (self.Nk1>0):
            self.P_converged=np.reshape(P_converged,(self.Nk0,self.Nk1))
        else:
            self.P_converged=np.reshape(P_converged,(self.Nk0,))

    def interpolate_P(self,use_P_fid=False):
        """
        typical use: interpolate a power spectrum binned sph/cyl to modes accessible by the box to modes of interest for the survey being forecast

        notes
        * default behaviour upon requesting extrapolation: 
          "ValueError: One of the requested xi is out of bounds in dimension 0"
        * if extrapolation is acceptable for your purposes:
          run with avoid_extrapolation=False
          (bounds_error supersedes fill_value, so there's no issue with 
          fill_value always being set to what it needs to be to permit 
          extrapolation [None for the nd case, "extrapolate" for the 1d case])
        """
        if use_P_fid:
            self.P_converged=self.P_fid
        else:
            if (self.P_converged is None):
                print("WARNING: P_converged DNE yet. \nAttempting to calculate it now...")
                self.avg_realizations()
            if (self.k0bins_interp is None):
                raise NotEnoughInfoError

        if (self.k1bins_interp is not None):
            kpar_have_lo=  self.k0bins[0]
            kpar_have_hi=  self.k0bins[-1]
            kperp_have_lo= self.k1bins[0]
            kperp_have_hi= self.k1bins[-1]

            kpar_want_lo=  self.k0bins_interp[0]
            kpar_want_hi=  self.k0bins_interp[-1]
            kperp_want_lo= self.k1bins_interp[0]
            kperp_want_hi= self.k1bins_interp[-1]

            if (kpar_want_lo<kpar_have_lo):
                extrapolation_warning("low kpar",   kpar_want_lo,  kpar_have_lo)
            if (kpar_want_hi>kpar_have_hi):
                extrapolation_warning("high kpar",  kpar_want_hi,  kpar_have_hi)
            if (kperp_want_lo<kperp_have_lo):
                extrapolation_warning("low kperp",  kperp_want_lo, kperp_have_lo)
            if (kperp_want_hi>kperp_have_hi):
                extrapolation_warning("high kperp", kperp_want_hi, kperp_have_hi)
            self.k0_interp_grid,self.k1_interp_grid=np.meshgrid(self.k0bins_interp,self.k1bins_interp,indexing="ij")
            self.P_interp=interpn((self.k0bins,self.k1bins),self.P_converged,(self.k0_interp_grid,self.k1_interp_grid),method=self.kind,bounds_error=self.avoid_extrapolation,fill_value=None)
        else:
            k_have_lo=self.k0bins[0]
            k_have_hi=self.k0bins[-1]
            k_want_lo=self.k0bins_interp[0]
            k_want_hi=self.k0bins_interp[-1]
            if (k_want_lo<k_have_lo):
                extrapolation_warning("low k",k_want_lo,k_have_lo)
            if (k_want_hi>k_have_hi):
                extrapolation_warning("high k",k_want_hi,k_have_hi)
            P_interpolator=interp1d(self.k0bins,self.P_converged,kind=self.kind,bounds_error=self.avoid_extrapolation,fill_value="extrapolate")
            self.P_interp=P_interpolator(self.k0bins_interp)

class per_antenna(object):
    def __init__(self,
                 mode="full",b_NS=8.5,b_EW=6.3,observing_dec=pi/60.,offset_deg=1.75*pi/180.,N_pert_types=4,
                 num_pbws_to_pert=0,pbw_pert_frac=1e-2,
                 num_timesteps=15,num_hrs=None,
                 nu_ctr=nu_HI_z0,
                 pbw_fidu=None
                 ):
        # array and observation geometry
        self.N_NS=N_NS_full
        self.N_EW=N_EW_full
        self.DRAO_lat=DRAO_lat
        if (mode=="pathfinder"):
            self.N_NS=self.N_NS//2
            self.N_EW=self.N_EW//2
        self.N_ant=self.N_NS*self.N_EW
        self.N_bl=self.N_ant*(self.N_ant-1)//2
        self.observing_dec=observing_dec
        self.num_timesteps=num_timesteps
        self.nu_ctr_MHz=nu_ctr
        self.nu_ctr_Hz=nu_ctr*1e6
        if (num_hrs is None):
            num_hrs=primary_beam_crossing_time(self.nu_ctr_Hz,dec=self.observing_dec,D=D) # freq needs to be in Hz
        self.num_hrs=num_hrs
        self.lambda_obs=c/self.nu_ctr_Hz
        if (pbw_fidu is None):
            pbw_fidu=self.lambda_obs/D
        self.pbw_fidu=pbw_fidu
        self.pbw_pert_frac=pbw_pert_frac
        
        # antenna positions xyz
        antennas_EN=np.zeros((self.N_ant,2))
        for i in range(self.N_NS):
            for j in range(self.N_EW):
                antennas_EN[i*self.N_EW+j,:]=[j*b_EW,i*b_NS]
        antennas_EN-=np.mean(antennas_EN,axis=0) # centre the Easting-Northing axes in the middle of the array
        offset=offset_deg*pi/180. # actual CHORD is not perfectly aligned to the NS/EW grid. Eyeballed angular offset.
        offset_from_latlon_rotmat=np.array([[np.cos(offset),-np.sin(offset)],[np.sin(offset),np.cos(offset)]]) # use this rotation matrix to adjust the NS/EW-only coords
        for i in range(self.N_ant):
            antennas_EN[i,:]=np.dot(antennas_EN[i,:].T,offset_from_latlon_rotmat)
        dif=antennas_EN[0,0]-antennas_EN[0,-1]+antennas_EN[0,-1]-antennas_EN[-1,-1]
        up=np.reshape(2+(-antennas_EN[:,0]+antennas_EN[:,1])/dif, (self.N_ant,1)) # eyeballed ~2 m vertical range that ramps ~linearly from a high near the NW corner to a low near the SE corner
        antennas_ENU=np.hstack((antennas_EN,up))
        
        zenith=np.array([np.cos(DRAO_lat),0,np.sin(DRAO_lat)]) # Jon math
        east=np.array([0,1,0])
        north=np.cross(zenith,east)
        lat_mat=np.vstack([north,east,zenith])
        antennas_xyz=antennas_ENU@lat_mat.T
        self.antennas_xyz=antennas_xyz

        pbw_types=np.zeros((self.N_ant,))
        self.N_pert_types=N_pert_types
        N_beam_types=N_pert_types+1
        self.N_beam_types=N_beam_types
        epsilons=np.zeros(N_beam_types)
        if (num_pbws_to_pert>0):
            epsilons[1:]=pbw_pert_frac*np.random.uniform(size=np.insert(N_pert_types,0,1))

            # the randomly drawn way (fallback)
            indices_of_ants_w_pert_pbws=np.random.randint(0,self.N_ant,size=num_pbws_to_pert) # indices of antenna pbs to perturb (independent of the indices of antenna positions to perturb, by design)
            pbw_types[indices_of_ants_w_pert_pbws]=np.random.randint(1,high=N_beam_types,size=np.insert(num_pbws_to_pert,0,1)) # leaves as zero the indices associated with unperturbed antennas
        
            # # the segmented-across-the-array way (hypothesized example of a case that should give k-dependent results)
            ### THIS HAS 100 PERTURBED ANTENNAS AND TWO PERTURBATION TYPES BAKED IN... SHOULD GENERALIZE IF I REALLY WANT TO ROLL WITH THIS!!
            ### + BAKE IN OTHER OPTIONS BEYOND JUST THE CORNERS
            # antenna_numbers= np.reshape(np.arange(self.N_ant),(self.N_NS,self.N_EW))# row-major array to take chunks of for the chunks to perturb
            # print("self.N_NS,self.N_EW=",self.N_NS,self.N_EW)
            # nw_corner_indices=np.reshape(antenna_numbers[:7,:7],(num_pbws_to_pert//2-1,)) # hackily hard-coding the two-class case I'm contrasting with on the 22 Oct 2025 beam meeting
            # se_corner_indices=np.reshape(antenna_numbers[self.N_NS-7:,self.N_EW-7:],(num_pbws_to_pert//2-1,))
            # nw_corner_indices=np.append(nw_corner_indices,7)
            # se_corner_indices=np.append(se_corner_indices,520)
            # np.savetxt("nw_corner_indices.txt",nw_corner_indices)
            # np.savetxt("se_corner_indices.txt",se_corner_indices)
            # pbw_types[nw_corner_indices]=1
            # pbw_types[se_corner_indices]=2
            # indices_of_ants_w_pert_pbws=np.concatenate((nw_corner_indices,se_corner_indices))
        else:
            indices_of_ants_w_pert_pbws=None
        self.pbw_types=pbw_types
        self.indices_of_ants_w_pert_pbws=indices_of_ants_w_pert_pbws
        self.epsilons=epsilons
        
        # ungridded instantaneous uv-coverage (baselines in xyz)        
        uvw_inst=np.zeros((self.N_bl,3))
        indices_of_constituent_ant_pb_types=np.zeros((self.N_bl,2))
        k=0
        for i in range(self.N_ant):
            for j in range(i+1,self.N_ant):
                uvw_inst[k,:]=antennas_xyz[i,:]-antennas_xyz[j,:]
                indices_of_constituent_ant_pb_types[k]=[pbw_types[i],pbw_types[j]] # 1/np.sqrt( ( (1/antenna_pbs[i]**2)+(1/antenna_pbs[j]**2) )/2. ) # this is for a simple Gaussian beam where the x- and y- FWHMs are the same. Once I get this version working, it should be straightforward to bump up the dimensions and add separate widths
                k+=1
        uvw_inst=np.vstack((uvw_inst,-uvw_inst))
        indices_of_constituent_ant_pb_types=np.vstack((indices_of_constituent_ant_pb_types,indices_of_constituent_ant_pb_types))
        self.uvw_inst=uvw_inst
        self.indices_of_constituent_ant_pb_types=indices_of_constituent_ant_pb_types
        
        # internal plotting suitable for two perturbation types
        # u_inst=uvw_inst[:,0]
        # v_inst=uvw_inst[:,1]
        # ad_hoc_antenna_colours=["r","y","b"]
        # plt.figure()
        # for i in range(N_beam_types):
        #     keep=np.nonzero(pbw_types==i)
        #     plt.scatter(antennas_xyz[keep,0],antennas_xyz[keep,1],c=ad_hoc_antenna_colours[i],label="antenna status "+str(i),edgecolors="k",lw=0.3,s=20)
        # plt.xlabel("x (m)")
        # plt.ylabel("y (m)")
        # plt.title("CHORD "+str(self.nu_ctr_MHz)+" MHz antenna positions by primary beam status")
        # plt.legend()
        # plt.savefig("ant_positions_colour_coded_by_ant_pert_status.png",dpi=350)
        
        # ant_a_type,ant_b_type=indices_of_constituent_ant_pb_types.T
        # print("len(ant_a_type)=",len(ant_a_type))
        # fig,axs=plt.subplots(2,3,figsize=(9,8))
        # num=0
        # ad_hoc_colours=["r","tab:orange","tab:purple","y","tab:green","b"] # still only valid for a two-perturbation test
        # for a in range(N_beam_types):
        #     for b in range(a,N_beam_types):
        #         keep=np.nonzero(np.logical_and(ant_a_type==a,ant_b_type==b))
        #         u_inst_ab=u_inst[keep]
        #         v_inst_ab=v_inst[keep]
        #         axs[num%2,num%3].scatter(u_inst_ab,v_inst_ab,c=ad_hoc_colours[num],label="antenna status "+str(a)+str(b),edgecolors="k",lw=0.15,s=4)
        #         axs[num%2,num%3].set_xlabel("u (λ)")
        #         axs[num%2,num%3].set_ylabel("v (λ)")
        #         axs[num%2,num%3].set_title("antenna status "+str(a)+str(b))
        #         axs[num%2,num%3].axis("equal")                
        #         num+=1
        # plt.suptitle("CHORD "+str(self.nu_ctr_MHz)+" MHz instantaneous uv coverage ")
        # plt.tight_layout()
        # plt.savefig("inst_uv_colour_coded_by_ant_pert_status.png",dpi=250)
        print("computed ungridded instantaneous uv-coverage")

        # rotation-synthesized uv-coverage *******(N_bl,3,N_timesteps), accumulating xyz->uvw transformations at each timestep
        hour_angle_ceiling=np.pi*num_hrs/12 # 2pi*num_hrs/24
        hour_angles=np.linspace(0,hour_angle_ceiling,num_timesteps)
        thetas=hour_angles*15*np.pi/180
        
        zenith=np.array([np.cos(self.observing_dec),0,np.sin(self.observing_dec)]) # Jon math redux
        east=np.array([0,1,0])
        north=np.cross(zenith,east)
        project_to_dec=np.vstack([east,north])

        uv_synth=np.zeros((2*self.N_bl,2,num_timesteps))
        for i,theta in enumerate(thetas): # thetas are the rotation synthesis angles (converted from hr. angles using 15 deg/hr rotation rate)
            accumulate_rotation=np.array([[ np.cos(theta),np.sin(theta),0],
                                        [-np.sin(theta),np.cos(theta),0],
                                        [ 0,            0,            1]])
            uvw_rotated=uvw_inst@accumulate_rotation
            uvw_projected=uvw_rotated@project_to_dec.T
            uv_synth[:,:,i]=uvw_projected/self.lambda_obs
        self.uv_synth=uv_synth
        print("synthesized rotation")

    def calc_dirty_image(self, Npix=1024, pbw_fidu_use=None,tol=1.75):
        if pbw_fidu_use is None: # otherwise, use the one that was passed
            pbw_fidu_use=self.pbw_fidu
        t0=time.time()
        uvmin=np.min([np.min(self.uv_synth[:,0,:]),np.min(self.uv_synth[:,1,:])])
        uvmax=np.max([np.max(self.uv_synth[:,0,:]),np.max(self.uv_synth[:,1,:])])
        uvbins=np.linspace(tol*uvmin,tol*uvmax,Npix)
        self.uvmin=uvmin
        self.uvmax=uvmax
        Npix=uvbins.shape[0]
        self.Npix=Npix
        uvmagmin=np.sort(np.abs(uvbins))[1]
        thetamax=1/uvmagmin # these are 1/-convention Fourier duals, not 2pi/-convention Fourier duals
        self.thetamax=thetamax
        d2u=uvbins[1]-uvbins[0]
        uubins,vvbins=np.meshgrid(uvbins,uvbins,indexing="ij")
        uvplane=0.*uubins
        uvbins_use=np.append(uvbins,uvbins[-1]+uvbins[1]-uvbins[0])
        pad_lo,pad_hi=get_padding(Npix)
        for i in range(self.N_beam_types):
            eps_i=self.epsilons[i]
            for j in range(i+1):
                eps_j=self.epsilons[j]
                here=(self.indices_of_constituent_ant_pb_types[:,0]==i)&(self.indices_of_constituent_ant_pb_types[:,1]==j) # which baselines to treat during this loop trip... pbws has shape (N_bl,2) ... one column for antenna a and the other for antenna b
                u_here=self.uv_synth[here,0,:] # [N_bl,3,N_hr_angles]
                v_here=self.uv_synth[here,1,:]
                N_bl_here,N_hr_angles_here=u_here.shape # (N_bl,N_hr_angles)
                N_here=N_bl_here*N_hr_angles_here
                reshaped_u=np.reshape(u_here,N_here)
                reshaped_v=np.reshape(v_here,N_here)
                gridded,_,_=np.histogram2d(reshaped_u,reshaped_v,bins=uvbins_use)
                width_here=pbw_fidu_use*np.sqrt((1-eps_i)*(1-eps_j))
                kernel=PA_Gaussian(uubins,vvbins,[0.,0.],width_here)
                kernel_padded=np.pad(kernel,((pad_lo,pad_hi),(pad_lo,pad_hi)),"edge") # version that worked in pipeline branch 2
                convolution_here=convolve(kernel_padded,gridded,mode="valid") # beam-smeared version of the uv-plane for this perturbation permutation
                uvplane+=convolution_here
        
        uvplane/=(self.N_beam_types**2*np.sum(uvplane)) # divide out the artifact of there having been multiple convolutions
        self.uvplane=uvplane
        dirty_image=np.abs(fftshift(ifft2(ifftshift(uvplane)*d2u,norm="forward")))
        dirty_image/=np.sum(dirty_image) # also account for renormalization in image space
        uv_bin_edges=[uvbins,uvbins]
        t1=time.time()
        self.dirty_image=dirty_image
        self.uv_bin_edges=uv_bin_edges
        return dirty_image,uv_bin_edges,thetamax

    def stack_to_box(self,delta_nu,evol_restriction_threshold=1./15., N_grid_pix=1024):
        if (self.nu_ctr_MHz<(350/(1-evol_restriction_threshold/2)) or self.nu_ctr_MHz>(nu_HI_z0/(1+evol_restriction_threshold/2))):
            raise SurveyOutOfBoundsError
        self.N_grid_pix=N_grid_pix
        bw_MHz=self.nu_ctr_MHz*evol_restriction_threshold
        N_chan=int(bw_MHz/delta_nu)
        self.nu_lo=self.nu_ctr_MHz-bw_MHz/2.
        self.nu_hi=self.nu_ctr_MHz+bw_MHz/2.
        surv_channels_MHz=np.linspace(self.nu_hi,self.nu_lo,N_chan) # decr.
        surv_channels_Hz=1e6*surv_channels_MHz
        surv_wavelengths=c/surv_channels_Hz # incr.
        surv_beam_widths=surv_wavelengths/D # incr.
        self.surv_channels=surv_channels_Hz
        self.z_channels=nu_HI_z0/surv_channels_MHz-1.
        self.comoving_distances_channels=np.asarray([comoving_distance(chan) for chan in self.z_channels]) # incr.
        self.ctr_chan_comov_dist=self.comoving_distances_channels[N_chan//2]

        box=np.zeros((N_grid_pix,N_grid_pix,N_chan))
        surv_beam_widths_desc=np.flip(surv_beam_widths) # traverse beam widths in descending order = handle first the slice with the narrowest uv bin extent
        for i,beam_width in enumerate(surv_beam_widths_desc):
            # rescale the uv-coverage to this channel's frequency
            self.uv_synth=self.uv_synth*self.lambda_obs/surv_wavelengths[i] # rescale according to observing frequency: multiply up by the prev lambda to cancel, then divide by the current/new lambda
            self.lambda_obs=surv_wavelengths[i] # update the observing frequency for next time

            # compute the dirty image
            chan_dirty_image,chan_uv_bin_edges,thetamax=self.calc_dirty_image(Npix=N_grid_pix, pbw_fidu_use=beam_width)
            
            # interpolate to store in stack
            if i==0:
                uv_bin_edges_0=chan_uv_bin_edges[0]
                uu_bin_edges_0,vv_bin_edges_0=np.meshgrid(uv_bin_edges_0,uv_bin_edges_0,indexing="ij")
                theta_max_box=thetamax
                interpolated_slice=chan_dirty_image
            else:
                # chunk excision and interpolation in one step:
                interpolated_slice=interpn(chan_uv_bin_edges,
                                           chan_dirty_image,
                                           (uu_bin_edges_0,vv_bin_edges_0),
                                           bounds_error=False, fill_value=None) # extrap necessary because the smallest u and v you have at a given slice-needing-extrapolation will be larger than the min u and v mags to extrapolate to
            box[:,:,i]=interpolated_slice
            if ((i%(N_chan//10))==0):
                print("{:5}%% complete".format(i/N_chan*100))
        self.box=box 
        self.theta_max_box=theta_max_box

        # generate a box of r-values (necessary for interpolation to survey modes in the manual beam mode of cosmo_stats as called by beam_effects)
        thetas=np.linspace(-self.theta_max_box,self.theta_max_box,N_grid_pix)
        xy_vec=self.ctr_chan_comov_dist*thetas # supply the thetas but multiply by the central channel's comoving distance (here, I'll need to make the coeval approximation)
        z_vec=self.comoving_distances_channels-self.ctr_chan_comov_dist # comoving distances from each freq channel... maybe eventually let cosmo_stats handle this? but also maybe not bc I call this before that? but also eventually I'll probably just refactor this to be a part of cosmo_stats?
        self.xy_vec=xy_vec
        self.z_vec=z_vec

# use the part of the Blues colour map with decent contrast and eyeball-ably perceivable differences between adjacent samplings
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=1000): # https://stackoverflow.com/questions/18926031/how-to-extract-a-subset-of-a-colormap-as-a-new-colormap-in-matplotlib
    new_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap
cmap = plt.get_cmap("Blues")
trunc_Blues = truncate_colormap(cmap, 0.2, 0.8)

def get_padding(n):
    padding=n-1
    padding_lo=int(np.ceil(padding / 2))
    padding_hi=padding-padding_lo
    return padding_lo,padding_hi

def PA_Gaussian(u,v,ctr,fwhm):
    u0,v0=ctr
    evaled=((pi*ln2)/(fwhm**2))*np.exp(-pi**2*(((u-u0)**2+(v-v0)**2)*fwhm**2)/np.log(2))
    # fwhmx,fwhmy=fwhm
    # evaled=((pi*ln2)/(fwhmx*fwhmy))*np.exp(-pi**2*((u-u0)**2*fwhmx**2+(v-v0)**2*fwhmy**2)/np.log(2))
    return evaled

def sparse_PA_Gaussian(u,v,ctr,fwhm,nsigma_npix):
    """
    same as the non-sparse version but uses scipy sparse arrays to make things less inefficient

    u,v  - square coordinate arrays defining the grid
    ctr  - uv coordinates of beam peak
    fwhm -  
    """
    # figure out where to put the Gaussian and its values
    u0,v0=ctr
    # will need indices of the peak of the beam in the uv plane for sparse array anchoring purposes
    base=0.*u
    evaled=((pi*ln2)/(fwhm**2))*np.exp(-pi**2*(((u-u0)**2+(v-v0)**2)*fwhm**2)/np.log(2))
    u0i,v0i=np.unravel_index(evaled.argmax(), evaled.shape)
    base[u0i-nsigma_npix:u0i+nsigma_npix,v0i-nsigma_npix:v0i+nsigma_npix]=evaled[u0i-nsigma_npix:u0i+nsigma_npix,v0i-nsigma_npix:v0i+nsigma_npix]
    evaled_sparse=spsp.csr_array(base)

    # mask the 10-sigma region and store as a sparse array
    return evaled_sparse

def primary_beam_crossing_time(nu,dec=30.,D=6.):
    beam_width_deg=1.029*(2.998e8/nu)/D*180/np.pi
    crossing_time_hrs_no_dec=beam_width_deg/15
    crossing_time_hrs= crossing_time_hrs_no_dec*np.cos(dec*pi/180)
    return crossing_time_hrs