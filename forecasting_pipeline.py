import numpy as np
import camb
from camb import model
from scipy.signal import convolve
from scipy.interpolate import interpn,interp1d
from scipy.special import j1
from numpy.fft import fftshift,ifftshift,fftn,irfftn,fftfreq
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
h_Planck18=H0_Planck18/100.
Omegamh2_Planck18=Omegam_Planck18*h_Planck18**2
pars_set_cosmo_Planck18=[H0_Planck18,Omegabh2_Planck18,Omegamh2_Planck18,AS_Planck18,ns_Planck18] # suitable for get_mps
infty=np.infty
pi=np.pi
twopi=2.*pi
ln2=np.log(2)
nu_rest_21=1420.405751768 # MHz
scale=1e-9
maxfloat= np.finfo(np.float64).max
huge=np.sqrt(maxfloat)
BasicAiryHWHM=1.616339948310703178119139753683896309743121097215461023581 # a preposterous number of sig figs from Mathematica (I haven't counted them but this is almost certainly overkill/ past the double-precision threshold)
eps=1e-15
maxfloat= np.finfo(np.float64).max
maxint=   np.iinfo(np.int64  ).max
nearly_zero=(1./maxfloat)**2

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
class ConflictingInfoError(Exception): # make these inherit from each other (or something) to avoid repetitive code
    pass


def extrapolation_warning(regime,want,have):
    print("WARNING: if extrapolation is permitted in the interpolate_P call, it will be conducted for {:15s} (want {:9.4}, have{:9.4})".format(regime,want,have))
    return None
def represent(cosmo_stats_instance):
    attributes=vars(cosmo_stats_instance)
    representation='\n'.join("%s: %s" % item for item in attributes.items())
    print(representation)

def Gaussian_primary(X,Y,Z,sigLoS,fwhm_x,fwhm_y,r0):
    """
    (Nvox,Nvox,Nvox) Cartesian box (z=LoS direction), centred at r0, sampling the response fcn at each point
    """
    return np.exp(-(Z/(2*sigLoS))**2 -ln2*((X/fwhm_x)**2+(Y/fwhm_y)**2)/r0**2)

def AiryGaussian_primary(X,Y,Z,sigLoS,fwhm_x,fwhm_y,r0):
    """
    (Nvox,Nvox,Nvox) Cartesian box (z=LoS direction), centred at r0, sampling the response fcn at each point
    """
    par= np.exp(-(Z/(2*sigLoS))**2)
    thetaX=X/r0
    argX=thetaX*BasicAiryHWHM/fwhm_x
    thetaY=Y/r0
    argY=thetaY*BasicAiryHWHM/fwhm_y
    perp=((j1(argX+eps)*j1(argY+eps))/((argX+eps)*(argY+eps)))**2
    return par*perp


"""
this module helps compute contaminant power and cosmological parameter biases using
a Fisher-based formalism using two complementary strategies with different scopes:
1. analytical windowing for a cylindrically symmetric Gaussian beam
2. numerical  windowing for a Gaussian beam with different x- and y-pol widths
"""

class window_calcs(object):
    def __init__(self,
                 bmin,bmax,                                             # extreme baselines of the array
                 ceil,                                                  # avoid kpars beyond the regime of linear theory
                 primary_beam_type,primary_beam_args,primary_beam_uncs, # primary beam considerations
                 pars_set_cosmo,pars_forecast,                          # implement soon: build out the functionality for pars_forecast to differ nontrivially from pars_set_cosmo
                 n_sph_modes,dpar,                                      # conditioning the CAMB/etc. call
                 nu_ctr,delta_nu,                                       # for the survey of interest
                 evol_restriction_threshold=1./15.,                     # misc. numerical considerations
                 init_and_box_tol=0.05,CAMB_tol=0.05,                   # considerations for k-modes at different steps
                 ftol_deriv=1e-6,eps=1e-16,maxiter=5,                   # precision control for numerical derivatives
                 uncs=None,frac_unc=0.1,                                # for Fisher-type calcs
                 Nkpar_box=15,Nkperp_box=18,frac_tol_conv=0.1,          # considerations for cyl binned power spectra from boxes
                 pars_forecast_names=None                               # for verbose output
                ):                                                      # implement soon: synthesized beam considerations, other primary beam types, and more
        """
        bmin,bmax                  :: floats                       :: max and min baselines of the array       :: m
        ceil                       :: int                          :: # high-kpar channels to ignore           :: ---
        primary_beam               :: callable                     :: power beam in Cartesian coords           :: ---
        primary_beam_type          :: str                          :: implement soon: Airy etc.                :: ---
        primary_beam_args          :: (N_args,) of floats          :: Gaussian, AiryGaussian: "μ"s and "σ"s    :: Gaussian: sigLoS, r0 in Mpc; fwhm_x, fwhm_y in rad
                                                                                PASS fwhm_x,fwhm_y
                                                                                ++sigLoS prepended internally
                                                                                ++r0 appended internally
        primary_beam_uncs          :: (N_uncertain_args) of floats :: Gaussian, AiryGaussian: fractional       :: ---
                                                                      uncertainties epsLoS, epsfwhmx, epsfwhmy 
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
                                      of strings     
        """
        # primary beam considerations
        if (primary_beam_type.lower()=="gaussian" or primary_beam_type.lower()=="airygaussian"):
            self.fwhm_x,self.fwhm_y=primary_beam_args
            self.primary_beam_uncs=              primary_beam_uncs
            self.epsLoS,self.epsx,self.epsy=     self.primary_beam_uncs
        else:
            raise NotYetImplementedError
        self.primary_beam_type=primary_beam_type
        self.primary_beam_args=primary_beam_args
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
        self.bw=nu_ctr*evol_restriction_threshold # previously called NDeltanu
        self.Nchan=int(self.bw/self.Deltanu)
        self.z_ctr=freq2z(nu_rest_21,nu_ctr)
        self.nu_lo=self.nu_ctr-self.bw/2.
        self.z_hi=freq2z(nu_rest_21,self.nu_lo)
        self.Dc_hi=comoving_distance(self.z_hi)
        self.nu_hi=self.nu_ctr+self.bw/2.
        self.z_lo=freq2z(nu_rest_21,self.nu_hi)
        self.Dc_lo=comoving_distance(self.z_lo)
        self.deltaz=self.z_hi-self.z_lo
        self.surv_channels=np.arange(self.nu_lo,self.nu_hi,self.Deltanu)
        if (primary_beam_type.lower()=="gaussian" or primary_beam_type.lower()=="airygaussian"):
            self.r0=comoving_distance(self.z_ctr)
            self.sigLoS=(self.r0-self.Dc_lo)/40.
            self.perturbed_primary_beam_args=(self.sigLoS*(1-self.epsLoS),self.fwhm_x*(1-self.epsx),self.fwhm_y*(1-self.epsy))
            self.primary_beam_args=np.array([self.sigLoS,self.fwhm_x,self.fwhm_y,self.r0]) # UPDATING ARGS NOW THAT THE FULL SET HAS BEEN SPECIFIED
            self.perturbed_primary_beam_args=np.append(self.perturbed_primary_beam_args,self.r0)
        else:
            raise NotYetImplementedError

        # cylindrically binned survey k-modes and box considerations
        kpar_surv=kpar(self.nu_ctr,self.Deltanu,self.Nchan)
        self.ceil=ceil
        self.kpar_surv=kpar_surv[:-self.ceil]
        self.Nkpar_surv=len(self.kpar_surv)
        self.bmin=bmin
        self.bmax=bmax
        self.kperp_surv=kperp(self.nu_ctr,self.Nchan,self.bmin,self.bmax)
        self.Nkperp_surv=len(self.kperp_surv)

        # self.kmin_surv=np.min((self.kpar_surv[0 ],self.kperp_surv[0 ])) # no extrap issues but slow (mult. factors of 1.1 and 1.2 also incur this issue, to their respective extents)
        self.kmin_surv=1.25*np.min((self.kpar_surv[ 0],self.kperp_surv[ 0])) # slow and slight extrap issues
        self.kmax_surv=np.sqrt(self.kpar_surv[-1]**2+self.kperp_surv[-1]**2) #very fast but extrap issues
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
        if (primary_beam_type.lower()=="gaussian" or primary_beam_type.lower()=="airygaussian"):
            sky_plane_sigmas=self.r0*np.array([self.fwhm_x,self.fwhm_y])/np.sqrt(2*np.log(2))
            self.all_sigmas=np.concatenate((sky_plane_sigmas,[self.sigLoS]))
            if (np.any(self.all_sigmas<self.Deltabox)):
                raise NumericalDeltaError
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

        pars_use_internal=camb.set_params(H0=H0, ombh2=ombh2, omch2=omch2, ns=ns, mnu=0.06,omk=0)
        pars_use_internal.InitPower.set_params(As=As,ns=ns,r=0)
        pars_use_internal.set_matter_power(redshifts=z, kmax=maxkh*h)
        results = camb.get_results(pars_use_internal)
        pars_use_internal.NonLinear = model.NonLinear_none
        kh,z,pk=results.get_matter_power_spectrum(minkh=minkh,maxkh=maxkh,npoints=self.n_sph_modes)
        return kh,pk
    
    def unbin_to_Pcyl(self,pars_to_use):
        """
        interpolate a spherically binned CAMB MPS to provide MPS values for a cylindrically binned k-grid of interest (nkpar x nkperp)
        """
        k,Psph_use=self.get_mps(pars_to_use,minkh=self.kmin_surv,maxkh=self.kmax_surv)
        Psph_use=Psph_use.reshape((Psph_use.shape[1],))
        kpar_grid,kperp_grid=np.meshgrid(self.kpar_surv,self.kperp_surv,indexing="ij")
        kmag_grid=np.sqrt(kpar_grid**2+kperp_grid**2)

        kmag_grid_flat=np.reshape(kmag_grid,(self.Nkpar_surv*self.Nkperp_surv,))
        Psph_interpolator=interp1d(k,Psph_use,kind="cubic",bounds_error=False,fill_value="extrapolate")
        P_interp_flat=Psph_interpolator(kmag_grid_flat)
        Pcyl=np.reshape(P_interp_flat,(self.Nkpar_surv,self.Nkperp_surv))
        return kpar_grid,kperp_grid,Pcyl

    def get_padding(self,n):
        padding=n-1
        padding_lo=int(np.ceil(padding / 2))
        padding_hi=padding-padding_lo
        return padding_lo,padding_hi
    
    def calc_Pcont_cyl(self):
        """
        calculate the cylindrically binned "contaminant power," following from the true and perceived window functions
        """
        self.calc_Wcont()
        if (self.Pcyl.shape!=self.Wcont.shape):
            if(self.Pcyl.shape.T!=self.Wcont.shape):
                assert(1==0), "window and power spec shapes must match"
            self.Wcont=self.Wcont.T # force P and Wcont to have the same shapes
        s0,s1=self.Pcyl.shape # by now, P and Wcont have the same shapes
        pad0lo,pad0hi=self.get_padding(s0)
        pad1lo,pad1hi=self.get_padding(s1)
        Wcontp=np.pad(self.Wcont,((pad0lo,pad0hi),(pad1lo,pad1hi)),"edge")
        conv=convolve(Wcontp,self.Pcyl,mode="valid")
        self.Pcont_cyl=conv ### same update as calc_Pcont_asym
    
    def W_cyl_binned(self,beam_pars_to_use):
        """
        wrapper to multiply the LoS and flat sky approximation sky plane terms of the cylindrically binned window function, for the grid described by the k-parallel and k-perp modes of the survey of interest
        """
        sigLoS_use,fwhm_x_use,_,r0_use=beam_pars_to_use
        par_vec=np.exp(-(self.kpar_surv*sigLoS_use)**2)
        perp_vec=np.exp(-(r0_use*fwhm_x_use*self.kperp_surv)**2/(2.*ln2))
        par_arr,perp_arr=np.meshgrid(par_vec,perp_vec,indexing="ij")
        meshed=par_arr*perp_arr
        raw_sum=np.sum(meshed)
        if raw_sum!=0.:
            return meshed/raw_sum
        else:
            return meshed

    def calc_Wcont(self):
        """
        calculate the "contaminant" windowing amplitude that will help give rise to the so-called "contaminant power"
        (ignores fwhmy and epsfwhmy because of the limits of analytical cylindrical math, although they need to be passed when initializing the class object to avoid unpacking errors)
        """
        self.Wtr=self.W_cyl_binned(self.primary_beam_args)
        self.Wth=self.W_cyl_binned(self.perturbed_primary_beam_args)
        self.Wcont=self.Wtr-self.Wth
        self.Wcontshape=self.Wcont.shape
    
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

        tr=cosmo_stats(self.Lsurvbox,
                       P_fid=self.Ptruesph,Nvox=self.Nvoxbox,
                       primary_beam=Gaussian_primary,primary_beam_args=self.primary_beam_args,
                       Nk0=self.Nkpar_box,Nk1=self.Nkperp_box,
                       frac_tol=self.frac_tol_conv,
                       k0bins_interp=self.kpar_surv,k1bins_interp=self.kperp_surv,
                       k_fid=self.ksph)
        th=cosmo_stats(self.Lsurvbox,
                       P_fid=self.Ptruesph,Nvox=self.Nvoxbox,
                       primary_beam=Gaussian_primary,primary_beam_args=self.perturbed_primary_beam_args,
                       Nk0=self.Nkpar_box,Nk1=self.Nkperp_box,
                       frac_tol=self.frac_tol_conv,
                       k0bins_interp=self.kpar_surv,k1bins_interp=self.kperp_surv,
                       k_fid=self.ksph)
        
        tr.avg_realizations()
        th.avg_realizations()

        self.Ptrue_cyl=    tr.P_converged
        self.Pthought_cyl= th.P_converged
        self.Pcont_cyl=    self.Ptrue_cyl-self.Pthought_cyl ### same update as calc_Pcont_sym

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
        if (self.fwhm_x!=self.fwhm_y):
            self.calc_Pcont_asym()
        else:
            self.calc_Pcont_cyl()
        print("computed Pcont")

        V=0.*self.cyl_partials
        for i in range(self.N_pars_forecast):
            V[i,:,:]=self.cyl_partials[i,:,:]/self.uncs # elementwise division for an nkpar x nkperp slice
        V_completely_transposed=np.transpose(V,axes=(2,1,0))
        F=np.einsum("ijk,kjl->il",V,V_completely_transposed)
        print("computed F")
        if (not np.all(self.Pcont_cyl.shape==self.uncs.shape)):
            interp_holder=cosmo_stats(self.Lsurvbox,P_fid=self.Pcont_cyl,Nvox=self.Nvoxbox,
                                      Nk0=self.Nkpar_box,Nk1=self.Nkperp_box,                                       
                                      k0bins_interp=self.kpar_surv,k1bins_interp=self.kperp_surv) # hacky use of interpolate_P means the Nk0- and Nk1-determined bins will be treated as fiducial (or, at least, that's what I need to make happen)
            interp_holder.interpolate_P(use_P_fid=True)
            self.Pcont_cyl_surv=interp_holder.P_interp
            print("interpolated Pcont to survey modes")
        else: # no interpolation necessary
            self.Pcont_cyl_surv=self.Pcont_cyl
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
        print("characteristic instrument response widths...............................................\n    sigLoS = {:>7.4}     Mpc (frac. uncert. {:>7.4})\n    beamFWHM = {:>=8.4}  rad (frac. uncert. {:>7.4})\n".format(self.sigLoS,self.epsLoS,self.fwhm_x,self.epsx))
        print("specific to the cylindrically asymmetric beam...........................................\n    beamFWHM1 {:>8.4} = rad (frac. uncert. {:>7.4}) \n".format(self.fwhm_y,self.epsy))
        print("cylindrically binned wavenumbers of the survey..........................................\n    kparallel {:>8.4} - {:>8.4} Mpc**(-1) ({:>4} channels of width {:>7.4}  Mpc**(-1)) \n    kperp     {:>8.4} - {:>8.4} Mpc**(-1) ({:>4} bins of width {:>8.4} Mpc**(-1))\n".format(self.kpar_surv[0],self.kpar_surv[-1],self.Nkpar_surv,self.kpar_surv[-1]-self.kpar_surv[-2],   self.kperp_surv[0],self.kperp_surv[-1],self.Nkperp_surv,self.kperp_surv[-1]-self.kperp_surv[-2]))
        print("cylindrically binned k-bin sensitivity..................................................\n    fraction of Pcyl amplitude = {:>7.4}".format(self.frac_unc))

    def print_results(self):
        print("\n\nbias calculation results for the survey described above.................................")
        print("........................................................................................")
        for p,par in enumerate(self.pars_forecast):
            print('{:12} = {:-10.3e} with bias {:-12.5e} (fraction = {:-10.3e})'.format(self.pars_forecast_names[p], par, self.biases[p], self.biases[p]/par))
        return None

"""
this module helps connect ensemble-averaged power spectrum estimates and 
cosmological brighness temperature boxes for assorted interconnected use cases:
1. generate a power spectrum that describes the statistics of a cosmo box
2. generate realizations of a cosmo box consistent with a known power spectrum
3. iterate power spec calcs from different box realizations until convergence
4. interpolate a power spectrum (sph, cyl, or sph->grid)
"""

class cosmo_stats(object):
    def __init__(self,
                 Lsurvey,                                                                # nonnegotiable for box->spec and spec->box calcs
                 T_pristine=None,T_primary=None,P_fid=None,Nvox=None,                    # need one of either T (pristine or primary) or P to get started; I also check for any conflicts with Nvox
                 primary_beam=None,primary_beam_args=None,primary_beam_type="Gaussian",  # primary beam considerations
                 Nk0=10,Nk1=0,binning_mode="lin",                                        # binning considerations for power spec realizations (log mode not fully tested yet b/c not impt. for current pipeline)
                 frac_tol=0.1,                                                           # max number of realizations
                 k0bins_interp=None,k1bins_interp=None,                                  # bins where it would be nice to know about P_converged
                 P_realizations=None,P_converged=None,                                   # power spectra related to averaging over those from dif box realizations
                 verbose=False,                                                          # status updates for averaging over realizations
                 k_fid=None,kind="cubic",avoid_extrapolation=False,                      # helper vars for converting a 1d fid power spec to a box sampling
                 no_monopole=False,                                                      # consideration when generating boxes
                 manual_primary_beam_modes=None                                          # when using a discretely sampled primary beam not sampled internally using a callable, it is necessary to provide knowledge of the modes at which it was sampled
                 ):                                                                      # implement soon: synthesized beam considerations, other primary beam types, and more
        """
        Lsurvey                   :: float                       :: side length of cosmo box         :: Mpc
        T_pristine                :: (Nvox,Nvox,Nvox) of floats  :: cosmo box (just physics/no beam) :: K
        T_primary                 :: (Nvox,Nvox,Nvox) of floats  :: cosmo box * primary beam         :: K
        P_fid                     :: (Nk0_fid,) of floats        :: sph binned fiducial power spec   :: K^2 Mpc^3
        Nvox                      :: float                       :: # voxels PER SIDE of cosmo box   :: ---
        primary_beam              :: callable                    :: power beam in Cartesian coords   :: ---
        primary_beam_args         :: tuple of floats             :: Gaussian, AiryGaussian: μ, σ     :: Gaussian: sigLoS, r0 in Mpc; fwhm_x, fwhm_y in rad
        primary_beam_type         :: str                         :: for now: Gaussian / AiryGaussian :: ---
        Nk0, Nk1                  :: int                         :: # power spec bins for axis 0/1   :: ---
        binning_mode              :: str                         :: lin/log sp. P_realizations bins  :: ---
        frac_tol                  :: float                       :: max fractional amount by which   :: ---
                                                                    the p.s. avg can change w/ the 
                                                                    addition of the latest realiz. 
                                                                    and the ensemble average is 
                                                                    considered converged
        k0bins_interp,            :: (Nk0_interp,) of floats     :: bins to which to interpolate the :: 1/Mpc
        k1bins_interp                (Nk1_interp,) of floats        converged power spec (prob set
                                                                    by survey considerations)
        P_realizations            :: if Nk1==0: (Nk0,)    floats :: sph/cyl power specs for dif      :: K^2 Mpc^3
                                     if Nk1>0:  (Nk0,Nk1) floats    realizations of a cosmo box 
        P_converged               :: same as that of P_fid       :: average of P_realizations        :: K^2 Mpc^3
        verbose                   :: bool                        :: every 10% of realization_ceil    :: ---
        k_fid                     :: (Nk0_fid,) of floats        :: modes where P_fid is sampled     :: 1/Mpc
        kind                      :: str                         :: interp type                      :: ---
        avoid_extrapolation       :: bool                        :: when calling scipy interpolators :: ---
        no_monopole               :: bool                        :: y/n subtr. from generated boxes  :: ---
        manual_primary_beam_modes :: primary_beam.shape of       :: domain of a discrete sampling    :: Mpc
                                     floats (when not callable) 
        """
        # spectrum and box
        self.Lsurvey=Lsurvey
        self.P_fid=P_fid
        self.T_primary=T_primary
        self.T_pristine=T_pristine
        self.no_monopole=no_monopole
        if ((T_primary is None) and (T_pristine is None) and (P_fid is None)):                           # must start with either a box or a fiducial power spec
            raise NotEnoughInfoError
        elif ((P_fid is not None) and (T_primary is None) and (T_pristine is None) and (Nvox is None)):  # to generate boxes from a power spec, il faut some way of determining how many voxels to use per side
            raise NotEnoughInfoError
        else:                                                           # there is possibly enough info to proceed, but still need to check for conflicts
            if ((T_pristine is not None) and (T_primary is not None)):
                print("WARNING: T_pristine and T_primary both passed; T_primary will be temporarily ignored and then internally overwritten to ensure consistency with primary_beam")
                if (T_pristine.shape!=T_primary.shape):
                    raise ConflictingInfoError
            if ((Nvox is not None) and (T_pristine is not None)):       # possible conflict: if both Nvox and a box are passed, 
                T_pristine_shape0=T_pristine.shape[0]
                if (Nvox!=T_pristine.shape[0]):                         # but Nvox and the box shape disagree,
                    raise ConflictingInfoError                          # estamos en problemas
                else:
                    self.Nvox=T_pristine_shape0                         # otherwise, initialize the Nvox attribute
            elif (Nvox is not None):                                    # if Nvox was passed but T was not, use Nvox to initialize the Nvox attribute
                self.Nvox=Nvox
            else:                                                       # remaining case: T was passed but Nvox was not, so use the shape of T to initialize the Nvox attribute
                self.Nvox=self.T_pristine_shape0
            
            if (P_fid is not None): # no hi fa res si the fiducial power spectrum has a different dimensionality or bin width than the realizations you plan to generate (boxes will be generated from a grid-interpolated P_fid, anyway)
                Pfidshape=P_fid.shape
                Pfiddims=len(Pfidshape)
                if (Pfiddims==2):
                    if primary_beam is None: # trying to do a minimalistic instantiation where I merely provide a fiducial power spectrum and interpolate it
                        self.fid_Nk0,self.fid_Nk1=Pfidshape
                    else:
                        try:
                            P_fid=np.reshape(P_fid,(Pfidshape[-1],)) # make the CAMB MPS shape amenable to the calcs internal to this class
                        except:
                            raise NotYetImplementedError
                elif (Pfiddims==1):
                    self.fid_Nk0=Pfidshape[0] # already checked that P_fid is 1d, so no info is lost by extracting the int in this one-element tuple, and fid_Nk0 being an integer makes things work the way they should down the line
                    self.fid_Nk1=0
                else:
                    raise UnsupportedBinningMode
        
        # config space
        self.Delta=self.Lsurvey/self.Nvox                            # voxel side length
        self.d3r=self.Delta**3                                       # volume element / voxel volume
        self.r_vec_for_box=self.Lsurvey*fftshift(fftfreq(self.Nvox))           # one Cartesian coordinate axis
        self.xx_grid,self.yy_grid,self.zz_grid=np.meshgrid(self.r_vec_for_box,
                                                           self.r_vec_for_box,
                                                           self.r_vec_for_box,
                                                           indexing="ij")      # box-shaped Cartesian coords
        self.r_grid=np.sqrt(self.xx_grid**2+self.yy_grid**2+self.zz_grid**2)   # r magnitudes at each voxel
        self.rmag_grid_centre_flat=np.reshape(self.r_grid,(Nvox**3,))

        # Fourier space
        self.Deltak=twopi/self.Lsurvey                                  # voxel side length
        self.d3k=self.Deltak**3                                         # volume element / voxel volume
        self.k_vec_for_box_corner=twopi*fftfreq(self.Nvox,d=self.Delta) # one Cartesian coordinate axis - non-fftshifted/ corner origin
        self.k_vec_for_box_centre=fftshift(self.k_vec_for_box_corner)   # one Cartesian coordinate axis -     fftshifted/ centre origin
        self.kx_grid_corner,self.ky_grid_corner,self.kz_grid_corner=np.meshgrid(self.k_vec_for_box_corner,
                                                                                self.k_vec_for_box_corner,
                                                                                self.k_vec_for_box_corner,
                                                                                indexing="ij")               # box-shaped Cartesian coords
        self.kmag_grid_corner= np.sqrt(self.kx_grid_corner**2+self.ky_grid_corner**2+self.kz_grid_corner**2) # k magnitudes for each voxel (need for the generate_box direction)
        self.kmag_grid_corner_flat=np.reshape(self.kmag_grid_corner,(self.Nvox**3,))
        self.kx_grid_centre,self.ky_grid_centre,self.kz_grid_centre=np.meshgrid(self.k_vec_for_box_centre,
                                                                                self.k_vec_for_box_centre,
                                                                                self.k_vec_for_box_centre,
                                                                                indexing="ij")
        # self.kmag_grid_centre=np.sqrt(self.kx_grid_centre**2+self.ky_grid_centre**2+self.kz_grid_centre**2)
        # self.kmag_grid_centre_flat=np.reshape(self.kmag_grid_corner,(self.Nvox**3,))

        # if P_fid was passed, establish its values on the k grid (helpful when generating a box)
        self.k_fid=k_fid
        self.kind=kind
        self.avoid_extrapolation=avoid_extrapolation
        if (self.P_fid is not None and self.k_fid is not None):
            try:
                self.P_fid=np.reshape(self.P_fid,self.P_fid.shape[1])
                self.P_fid_interp_1d_to_3d()
            except:
                print("no gridded P_fid was calculated because a 2D P_fid was passed")
        # no gridded P_fid is established when a 2D P_fid is passed (this should only be when the only computation I'm interested in doing is an interpolation of a cyl binned power spec that I pass as P_fid for convenience)
        
        # binning considerations
        self.binning_mode=binning_mode
        self.Nk0=Nk0 # the number of bins to put in power spec realizations you construct (ok if not the same as the number of bins in the fiducial power spec)
        self.Nk1=Nk1
        self.kmax_box=   pi/self.Delta
        self.kmin_box=twopi/self.Lsurvey
        self.k0bins,self.limiting_spacing_0=self.calc_bins(self.Nk0)
        if self.limiting_spacing_0<self.Deltak: # trying to bin more finely than the box can tell you about (guaranteed to have >=1 empty bin)
            raise ResolutionError
        
        if (self.Nk1>0):
            self.k1bins,self.limiting_spacing_1=self.calc_bins(self.Nk1)
            if (self.limiting_spacing_1<self.Deltak): # idem ^
                raise ResolutionError
        else:
            self.k1bins=None
        
            # voxel grids for sph binning
        self.sph_bin_indices_corner=      np.digitize(self.kmag_grid_corner,self.k0bins)                                  # sph bin that each voxel falls into
        self.sph_bin_indices_1d_corner=   np.reshape(self.sph_bin_indices_corner, (self.Nvox**3,)) # 1d version of ^ (compatible with np.bincount)
        
        self.k_grid_centre=               np.sqrt(self.kx_grid_centre**2+self.ky_grid_centre**2+self.kz_grid_centre**2)   # same thing but fftshifted/ centre-origin (need for the generate_P direction)
        self.sph_bin_indices_centre=      np.digitize(self.k_grid_centre,self.k0bins)
        self.sph_bin_indices_1d_centre=   np.reshape(self.sph_bin_indices_centre, (self.Nvox**3,))

            # voxel grids for cyl binning
        if (self.Nk1>0):
            self.kpar_column_centre= np.abs(self.k_vec_for_box_centre)                                          # magnitudes of kpar for a representative column along the line of sight (z-like)
            self.kperp_slice_centre= np.sqrt(self.kx_grid_centre**2+self.ky_grid_centre**2)[:,:,0]              # magnitudes of kperp for a representative slice transverse to the line of sight (x- and y-like)
            self.perpbin_indices_slice_centre=    np.digitize(self.kperp_slice_centre,self.k1bins)              # cyl kperp bin that each voxel falls into
            self.perpbin_indices_slice_1d_centre= np.reshape(self.perpbin_indices_slice_centre,(self.Nvox**2,)) # 1d version of ^ (compatible with np.bincount)
            self.parbin_indices_column_centre=    np.digitize(self.kpar_column_centre,self.k0bins)              # cyl kpar bin that each voxel falls into

        # primary beam
        self.primary_beam=primary_beam
        self.primary_beam_args=primary_beam_args
        self.primary_beam_type=primary_beam_type
        self.manual_primary_beam_modes=manual_primary_beam_modes
        if (self.primary_beam is not None): # non-identity primary beam
            if (self.primary_beam_type=="Gaussian" or self.primary_beam_type=="AiryGaussian"):
                self.sigLoS,self.fwhm_x,self.fwhm_y,self.r0=self.primary_beam_args
                evaled_primary=self.primary_beam(self.xx_grid,self.yy_grid,self.zz_grid,self.sigLoS,self.fwhm_x,self.fwhm_y,self.r0)
            elif (self.primary_beam_type=="manual"):
                if (self.primary_beam_args is not None):
                    raise ConflictingInfoError
                try:    # to access this branch, the manual/ numerically sampled primary beam needs to be close enough to a numpy array that it has a shape and not, e.g. a callable
                    primary_beam.shape
                except: # primary beam is a callable (or something else without a shape method), which is not in line with how this part of the code is supposed to work
                    raise ConflictingInfoError 
                if self.manual_primary_beam_modes is None:
                    raise NotEnoughInfoError
                beam_interpolator=interp1d(self.manual_primary_beam_modes,primary_beam,kind=self.kind,bounds_error=self.avoid_extrapolation,fill_value="extrapolate")
                interpolated_primary_beam=beam_interpolator(self.rmag_grid_centre_flat)
                interpolated_primary_beam=np.reshape(interpolated_primary_beam,(Nvox,Nvox,Nvox))
                self.primary_beam=interpolated_primary_beam # ok to store in class attribute now bc response has been interpolated to box modes (where it is necessary to know about the primary beam values to proceed)
            else:
                raise NotYetImplementedError
            self.Veff=np.sum(evaled_primary*self.d3r)        # rectangular sum method
            evaled_primary[evaled_primary<nearly_zero]=maxfloat # protect against division-by-zero errors
            self.evaled_primary=evaled_primary
        else:                               # identity primary beam
            self.Veff=self.Lsurvey**3
            self.evaled_primary=np.ones((self.Nvox,self.Nvox,self.Nvox))
        if (self.T_pristine is not None):
            self.T_primary=self.T_pristine*self.evaled_primary
        
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

    def calc_bins(self,Nki):
        """
        philosophy: generate a set of bins spaced according to the desired scheme with max and min
        """
        if (self.binning_mode=="log"):
            kbins=np.logspace(np.log10(self.kmin_box),np.log10(self.kmax_box),num=Nki)
            limiting_spacing=twopi*(10.**(self.kmax_box)-10.**(self.kmax_box-(np.log10(self.Nvox)/Nki)))
        elif (self.binning_mode=="lin"):
            kbins=np.linspace(self.kmin_box,self.kmax_box,Nki)
            limiting_spacing=twopi*(0.5*self.Nvox-1)/(Nki) # version for a kmax that is "aware that" there are +/- k-coordinates in the box
        else:
            raise UnsupportedBinningMode
        return kbins,limiting_spacing # kbins            -> floors of the bins to which the power spectrum will be binned (along one axis)
                                      # limiting_spacing -> smallest spacing between adjacent bins (uniform if linear; otherwise, depends on the binning strategy)
    
    def P_fid_interp_1d_to_3d(self):
        """
        interpolate a "physics-only" (spherically symmetric) power spectrum (e.g. from CAMB) to a 3D cosmological box.
        for now, I don't have a solution better than overwriting the k=0 term after the fact (because extrapolation to this term based on a reasonable CAMB call leads to negative power there)
        """
        P_fid_interpolator=interp1d(self.k_fid,self.P_fid,kind=self.kind,bounds_error=self.avoid_extrapolation,fill_value="extrapolate")
        P_interp_flat=P_fid_interpolator(self.kmag_grid_corner_flat)
        self.P_fid_box=np.reshape(P_interp_flat,(self.Nvox,self.Nvox,self.Nvox))
            
    def generate_P(self,send_to_P_fid=False):
        """
        philosophy: 
        * compute the power spectrum of a known cosmological box and bin it spherically or cylindrically
        * append to the list of reconstructed P realizations (self.P_realizations)
        """
        if (self.T_pristine is None):    # power spec has to come from a box
            self.generate_box() # populates/overwrites self.T_pristine and self.T_primary

        T_tilde=            fftshift(fftn((ifftshift(self.T_pristine)*self.d3r)))
        modsq_T_tilde=     (T_tilde*np.conjugate(T_tilde)).real
        modsq_T_tilde[:,:,self.Nvox//2]*=2 # attempt to fix the line of sight stripe numerically (study the formerly stripy plot) (LOOKS OKAY FOR ODD NVOX/SIDE) (there is a corner-L that remains stripy this way for EVEN Nvox/side, but that follows from the math iirc)

        if (self.Nk1==0):   # bin to sph
            modsq_T_tilde_1d= np.reshape(modsq_T_tilde,    (self.Nvox**3,))

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
            for i in range(self.Nvox): # iterate over the kpar axis of the box to capture all LoS slices
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

        avg_modsq_T_tilde=sum_modsq_T_tilde_truncated/N_modsq_T_tilde_truncated # actual estimator math
        P=np.array(avg_modsq_T_tilde/self.Veff)
        P.reshape(final_shape)
        if send_to_P_fid: # if generate_P was called speficially to have a spec from which all future box realizations will be generated
            self.P_fid=P
            self.P_fid_interp_1d_to_3d() # generate interpolated values of the newly established 1D P_fid over the k-magnitudes of the box
        else:             # the "normal" case where you're just accumulating a realization
            self.P_realizations.append([P])
        self.unbinned_P=modsq_T_tilde/self.Veff # box-shaped, but calculated according to the power spectrum estimator equation
        
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
        if (self.Veff<0):
            raise PathologicalError
        if (np.min(self.P_fid_box)<0):
            self.P_fid_box[self.P_fid_box<0]=0 # hackily overwriting error from having to extrapolate at the origin
        sigmas=np.sqrt(self.Veff*self.P_fid_box/2.) # from inverting the estimator equation and turning variances into std devs
        T_tilde_Re,T_tilde_Im=np.random.normal(loc=0.*sigmas,scale=sigmas,size=np.insert(sigmas.shape,0,2))
        
        T_tilde=T_tilde_Re+1j*T_tilde_Im # have not yet applied the symmetry that ensures T is real-valued 
        T=fftshift(irfftn(T_tilde*self.d3k,s=(self.Nvox,self.Nvox,self.Nvox),axes=(0,1,2),norm="forward"))/(twopi)**3 # handle in one line: fftshiftedness, ensuring T is real-valued and box-shaped, enforcing the cosmology Fourier convention
        if self.no_monopole:
            T-=np.mean(T) # subtract monopole moment to make things more akin to what powerbox does
        self.T_pristine=T 
        self.T_primary=T*self.evaled_primary

    def avg_realizations(self):
        assert(self.P_fid is not None), "cannot average over numerically windowed realizations without a fiducial power spec"
        self.not_converged=True
        i=0
        for i in range(self.realization_ceiling):
            self.generate_box()
            self.generate_P()
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