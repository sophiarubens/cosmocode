import numpy as np
import camb
from camb import model
from scipy.signal import convolve2d,convolve
from matplotlib import pyplot as plt
from power_class import *
from cosmo_distances import *
import time

"""
this module helps compute contaminant power and cosmological parameter biases using
a Fisher-based formalism using two complementary strategies with different scopes:
1. analytical windowing for a cylindrically symmetric Gaussian beam
2. numerical  windowing for a Gaussian beam with different x- and y-pol widths
"""

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
ln2=np.log(2)
nu_rest_21=1420.405751768 # MHz
scale=1e-9
maxfloat= np.finfo(np.float64).max
huge=np.sqrt(maxfloat)

h_Planck18=H0_Planck18/100.
Omegamh2_Planck18=Omegam_Planck18*h_Planck18**2
pars_set_cosmo_Planck18=[H0_Planck18,Omegabh2_Planck18,Omegamh2_Planck18,AS_Planck18,ns_Planck18] # suitable for get_mps

class NotYetImplementedError(Exception):
    pass
class PathologicalError(Exception):
    pass
class NvoxPracticalityError(Exception):
    pass
class NumericalDeltaError(Exception):
    pass

def Gaussian_primary(X,Y,Z,sigLoS,fwhm_x,fwhm_y,r0):
    """
    (Nvox,Nvox,Nvox) Cartesian box (z=LoS direction), centred at r0, sampling the response fcn at each point
    """
    return np.exp(-(Z/(2*sigLoS))**2 -ln2*((X/fwhm_x)**2+(Y/fwhm_y)**2)/r0**2)

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
        primary_beam_args          :: (N_args,) of floats          :: Gaussian: "μ"s and "σ"s                  :: Gaussian: sigLoS, r0 in Mpc; fwhm_x, fwhm_y in rad
                                                                                PASS sigLoS,fwhm_x,fwhm_y
                                                                                ++ r0 appended internally
        primary_beam_uncs          :: (N_uncertain_args) of floats :: Gaussian: frac uncs epsLoS, epsfwhm{x/y} :: ---
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
        if (primary_beam_type.lower()=="gaussian"):
            self.sigLoS,self.fwhm_x,self.fwhm_y= primary_beam_args # AS PASSED; APPEND R0 AFTER CALCULATING IT
            self.primary_beam_uncs=              primary_beam_uncs
            self.epsLoS,self.epsx,self.epsy=     self.primary_beam_uncs
            self.perturbed_primary_beam_args=(self.sigLoS*(1-self.epsLoS),self.fwhm_x*(1-self.epsx),self.fwhm_y*(1-self.epsy))
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
        self.r0=comoving_distance(self.z_ctr)
        if (primary_beam_type.lower()=="gaussian"):
            self.primary_beam_args=np.array([self.sigLoS,self.fwhm_x,self.fwhm_y,self.r0]) # UPDATING ARGS NOW THAT THE FULL SET HAS BEEN SPECIFIED
            self.perturbed_primary_beam_args=np.append(self.perturbed_primary_beam_args,self.r0)
        else:
            raise NotYetImplementedError
        self.nu_lo=self.nu_ctr-self.bw/2.
        self.z_hi=freq2z(nu_rest_21,self.nu_lo)
        self.Dc_hi=comoving_distance(self.z_hi)
        self.nu_hi=self.nu_ctr+self.bw/2.
        self.z_lo=freq2z(nu_rest_21,self.nu_hi)
        self.Dc_lo=comoving_distance(self.z_lo)
        self.deltaz=self.z_hi-self.z_lo
        self.surv_channels=np.arange(self.nu_lo,self.nu_hi,self.Deltanu)

        # cylindrically binned survey k-modes and box considerations
        kpar_surv=kpar(self.nu_ctr,self.Deltanu,self.Nchan)
        self.ceil=ceil
        self.kpar_surv=kpar_surv[:-self.ceil]
        self.Nkpar_surv=len(self.kpar_surv)
        self.deltakpar_surv=  self.kpar_surv  - self.kpar_surv[ self.Nkpar_surv// 2] 
        self.bmin=bmin
        self.bmax=bmax
        self.kperp_surv=kperp(self.nu_ctr,self.Nchan,self.bmin,self.bmax)
        self.Nkperp_surv=len(self.kperp_surv)
        self.deltakperp_surv= self.kperp_surv - self.kperp_surv[self.Nkperp_surv//2]

        self.kmin_surv=np.sqrt(self.kpar_surv[ 0]**2+self.kperp_surv[ 0]**2)
        self.kmax_surv=np.sqrt(self.kpar_surv[-1]**2+self.kperp_surv[-1]**2)
        self.Lsurvbox= twopi/self.kmin_surv                  ### CHECK THAT NO FACTOR OF TWO IS MISSING
        self.Nvoxbox=  int(self.Lsurvbox*self.kmax_surv/pi)
        self.NvoxPracticalityWarning()

        # numerical protections for assorted k-ranges
        limiting_spacing_surv=np.min([ self.kpar_surv[1]-self.kpar_surv[0], self.kperp_surv[1]-self.kperp_surv[0]])
        kmin_box_and_init=(1-init_and_box_tol)*self.kmin_surv
        kmax_box_and_init=(1+init_and_box_tol)*self.kmax_surv
        kmin_CAMB=(1-CAMB_tol)*kmin_box_and_init
        kmax_CAMB=(1+CAMB_tol)*kmax_box_and_init
        self.ksph,self.Ptruesph=self.get_mps(self.pars_set_cosmo,kmin_CAMB,kmax_CAMB*np.sqrt(3)) # factor of sqrt(3) from pythag theorem for box to prevent the need for extrap
        limiting_spacing_CAMB_sm=self.ksph[1]-self.ksph[0]
        limiting_spacing_CAMB_lg=self.ksph[-1]-self.ksph[-2]
        limiting_spacing_box_sm=2./(self.Nvoxbox*self.Lsurvbox*np.sqrt(3)*((np.sqrt(3)/2)-(1./self.Nvoxbox)))
        limiting_spacing_box_lg=self.Nvoxbox/(2.*self.Lsurvbox)
        self.Deltabox=self.Lsurvbox/self.Nvoxbox
        if primary_beam_type.lower()=="gaussian":
            sky_plane_sigmas=self.r0*np.array([self.fwhm_x,self.fwhm_y])/np.sqrt(2*np.log(2))
            self.all_sigmas=np.concatenate((sky_plane_sigmas,[self.sigLoS]))
            if (np.any(self.all_sigmas<self.Deltabox)):
                raise NumericalDeltaError
        else:
            raise NotYetImplementedError
        print("\nLIMITING SPACINGS")
        print("limiting_spacing_CAMB=",limiting_spacing_CAMB_sm,"-",limiting_spacing_CAMB_lg)
        print("limiting_spacing_box= ",limiting_spacing_box_sm,"-",limiting_spacing_box_lg)
        print("limiting_spacing_surv=",limiting_spacing_surv)

        print("\nMINS AND MAXES:")
        print("CAMB:           ",self.ksph[0],"-",self.ksph[-1])
        print("box vec:         I jump straight to the r-grid so this doesn't live in a variable thus far")
        # print("init par & perp: ",self.kpartrue_init[0],"-",self.kpartrue_init[-1],"&",self.kperptrue_init[0],"-",self.kperptrue_init[-1])
        print("surv par & perp: ",self.kpar_surv[0],"-",self.kpar_surv[-1],"&",self.kperp_surv[0],"-",self.kperp_surv[-1])
        
        # considerations for power spectra binned to survey k-modes
        _,_,self.Pcyl=self.unbin_to_Pcyl(self.pars_set_cosmo) # unbin_to_Pcyl(self,pars_to_use)
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
        self.Pcont_cyl=convolve(Wcontp,self.Pcyl,mode="valid") ### same update as calc_Pcont_asym
    
    def W_cyl_binned(self,beam_pars_to_use):
        """
        wrapper to multiply the LoS and flat sky approximation sky plane terms of the cylindrically binned window function, for the grid described by the k-parallel and k-perp modes of the survey of interest
        """
        sigLoS_use,fwhm_x_use,_,r0_use=beam_pars_to_use
        print("sigLoS_use,fwhm_x_use,r0_use=")
        par_vec=np.exp(-(self.deltakpar_surv*sigLoS_use)**2)
        perp_vec=np.exp(-(r0_use*fwhm_x_use*self.deltakperp_surv)**2/(2.*ln2))
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
        # self.Wth=self.W_cyl_binned(self.perturbed_primary_beam_args)
        self.Wth=0.*self.Wtr
        print("next line will store Wcont in self.Wcont")
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
        # th.avg_realizations()

        self.Ptrue_cyl=    tr.P_converged
        self.Pthought_cyl=0.*self.Ptrue_cyl
        # self.Pthought_cyl= th.P_converged
        self.Pcont_cyl=    self.Ptrue_cyl-self.Pthought_cyl ### same update as calc_Pcont_sym

    def cyl_partial(self,n):  
        """        
        cylindrically binned matter power spectrum partial WRT one cosmo parameter (nkpar x nkperp)
        """
        dparn=self.dpar[n]
        # print("self.iter,dparn=",self.iter,dparn)
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
        print("np.any(np.isnan(self.cyl_partials))=",np.any(np.isnan(self.cyl_partials)))
        print("built partials")
        if (self.fwhm_x!=self.fwhm_y):
            self.calc_Pcont_asym()
        else:
            self.calc_Pcont_cyl()
        print("np.any(np.isnan(self.Pcont_cyl))=",np.any(np.isnan(self.Pcont_cyl)))
        print("computed Pcont")

        V=0.*self.cyl_partials
        for i in range(self.N_pars_forecast):
            V[i,:,:]=self.cyl_partials[i,:,:]/self.uncs # elementwise division for an nkpar x nkperp slice
        V_completely_transposed=np.transpose(V,axes=(2,1,0))
        F=np.einsum("ijk,kjl->il",V,V_completely_transposed)
        print("np.any(np.isnan(F))=",np.any(np.isnan(F)))
        print("computed F")
        if (not np.all(self.Pcont_cyl.shape==self.uncs.shape)):
            interp_holder=cosmo_stats(self.Lsurvbox,P_fid=self.Pcont_cyl,Nvox=self.Nvoxbox,
                                      Nk0=self.Nkpar_box,Nk1=self.Nkperp_box,                                       
                                      k0bins_interp=self.kpar_surv,k1bins_interp=self.kperp_surv) # hacky use of interpolate_P means the Nk0- and Nk1-determined bins will be treated as fiducial (or, at least, that's what I need to make happen)
            interp_holder.interpolate_P(use_P_fid=True)
            self.Pcont_cyl_surv=interp_holder.P_interp
            print("np.any(np.isnan(self.Pcont_cyl_surv))=",np.any(np.isnan(self.Pcont_cyl_surv)))
            print("interpolated Pcont to survey modes")
        else: # no interpolation necessary
            self.Pcont_cyl_surv=self.Pcont_cyl
        Pcont_div_sigma=self.Pcont_cyl_surv/self.uncs
        print("np.any(np.isnan(Pcont_div_sigma))=",np.any(np.isnan(Pcont_div_sigma)))
        B=np.einsum("jk,ijk->i",Pcont_div_sigma,V)
        print("np.any(np.isnan(B))=",np.any(np.isnan(B)))
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