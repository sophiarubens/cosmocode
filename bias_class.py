import numpy as np
import camb
from camb import model
from scipy.signal import convolve2d,convolve
from matplotlib import pyplot as plt
from power_class import *
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
    return np.exp(-(Z/(2*sigLoS))**2 -ln2*((X/fwhm_x)**2+(Y/fwhm_y)**2)/r0**2)

class window_calcs(object):
    def __init__(self,
                 kpar_surv,kperp_surv,                                  # set by survey properties
                 primary_beam_type,primary_beam_args,primary_beam_uncs, # primary beam considerations
                 pars_set_cosmo,pars_forecast,                          # eventually: build out the functionality for pars_forecast to differ nontrivially from pars_set_cosmo
                 z,n_sph_modes,dpar,                                    # conditioning the CAMB/etc. call
                 nu_ctr,delta_nu,                                       # for the survey of interest
                 evol_restriction_threshold=1./15.,                     # misc. numerical considerations
                 init_and_box_tol=0.05,CAMB_tol=0.05                    # considerations for k-modes at different steps
                ):
        """
        kpar_surv,kperp_surv       :: (Nkpar,),(Nkperp,) of floats :: mono incr. cyl binned curvey modes       :: 1/Mpc
        primary_beam               :: callable                     :: power beam in Cartesian coords           :: ---
        primary_beam_type          :: str                          :: implement soon: Airy etc.                :: ---
        primary_beam_args          :: (N_args,) of floats          :: Gaussian: "μ"s and "σ"s                  :: Gaussian: sigLoS, r0 in Mpc; fwhm_x, fwhm_y in rad
                                                                                sigLoS,fwhm_x,fwhm_y,r0
        primary_beam_uncs          :: (N_uncertain_args) of floats :: Gaussian: frac uncs epsLoS, epsfwhm{x/y} :: ---
        pars_set_cosmo             :: (N_fid_pars,) of floats      :: params to condition a CAMB/etc. call     :: as found in ΛCDM
        pars_forecast              :: (N_forecast_pars,) of floats :: params for which you'd like to forecast  :: as found in ΛCDM
        z                          :: float                        :: z of fiducial MPS for CAMB/etc. call     :: ---
        n_sph_modes                :: int                          :: # modes to put in CAMB/etc. MPS          :: ---
        dpar                       :: (N_forecast_pars,) of floats :: initial guess of num. dif. step sizes    :: same as for pars_forecast
        nu_ctr                     :: float                        :: central freq for survey of interest      :: MHz
        delta_nu                   :: float                        :: channel width for survey of interest     :: MHz
        evol_restriction_threshold :: float                        :: ~$\frac{\Delta z}{z}$ w/in survey box    :: ---
        init_and_box_tol, CAMB_tol :: floats                       :: how much wider do you want the k-ranges  :: ---
                                                                      of preceding steps to be? (frac tols)
        """
        # cylindrically binned survey k-modes and box considerations
        self.kpar_surv=kpar_surv
        self.Nkpar_surv=len(kpar_surv)
        self.deltakpar_surv=  self.kpar_surv  - self.kpar_surv[ self.Nkpar_surv// 2] 
        self.kperp_surv=kperp_surv
        self.Nkperp_surv=len(kperp_surv)
        self.deltakperp_surv= self.kperp_surv - self.kperp_surv[self.Nkperp_surv//2]

        self.kmin_surv=np.sqrt(kpar_surv[ 0]**2+kperp_surv[ 0]**2)
        self.kmax_surv=np.sqrt(kpar_surv[-1]**2+kperp_surv[-1]**2)
        self.Lsurvbox= twopi/self.kmin                  ### CHECK THAT NO FACTOR OF TWO IS MISSING
        self.Nvoxbox=  int(self.Lsurvbox*self.kmax/pi)
        self.NvoxPracticalityWarning(self.Nvoxbox)

        # primary beam considerations
        if (primary_beam_type.lower()=="gaussian"):
            self.sigLoS,self.fwhm_x,self.fwhm_y,self.r0= self.primary_beam_args
            self.epsLoS,self.epsx,self.epsy=             self.primary_beam_uncs
            self.perturbed_primary_beam_args=(self.sigLoS*(1-self.epsLoS),self.fwhm_x*(1-self.epsx),self.fwhm_y*(1-self.epsy),self.r0)
        else:
            raise NotYetImplementedError
        self.primary_beam_type=primary_beam_type
        self.primary_beam_args=primary_beam_args
        self.primary_beam_uncs=primary_beam_uncs
        
        # forecasting considerations
        self.pars_set_cosmo=pars_set_cosmo
        self.pars_forecast=pars_forecast
        self.z=z
        self.n_sph_modes=n_sph_modes
        self.dpar=dpar
        self.nu_ctr=nu_ctr
        self.Deltanu=delta_nu
        self.bw=nu_ctr*evol_restriction_threshold # previously called NDeltanu
        self.Nchan=int(self.bw/self.Deltanu)

        # numerical protections for assorted k-ranges
        limiting_spacing_surv=np.min([ self.kpar[1]-self.kpar[0], self.kperp[1]-self.kperp[0]])
        kmin_box_and_init=(1-init_and_box_tol)*self.kmin_surv
        kmax_box_and_init=(1+init_and_box_tol)*self.kmax_surv
        kmin_CAMB=(1-CAMB_tol)*kmin_box_and_init
        kmax_CAMB=(1+CAMB_tol)*kmax_box_and_init
        self.ksph,self.Ptruesph=self.get_mps(kmin_CAMB,kmax_CAMB)
        limiting_spacing_CAMB_sm=self.ksph[1]-self.ksph[0]
        limiting_spacing_CAMB_lg=self.ksph[-1]-self.ksph[-2]
        limiting_spacing_box_sm=2./(self.Nvoxbox*self.Lsurvbox*np.sqrt(3)*((np.sqrt(3)/2)-(1./self.Nvoxbox)))
        limiting_spacing_box_lg=self.Nvoxbox/(2.*self.Lsurvbox)
        self.Deltabox=self.Lsurvbox/self.Nvoxbox
        if primary_beam_type.lower()=="gaussian":
            self.sky_plane_sigmas=self.r0*np.array([self.beamfwhm_x,self.beamfwhm_y])/np.sqrt(2*np.log(2))
            self.all_sigmas=np.concatenate((self.sky_plane_sigmas,[self.sigLoS]))
            if (np.any(self.all_sigmas<self.Deltabox)):
                raise NumericalDeltaError
        else:
            raise NotYetImplementedError
        print("END OF ITERATION 0 - COMPARISONS")
        print("\nLIMITING SPACINGS")
        print("limiting_spacing_CAMB=",limiting_spacing_CAMB_sm,"-",limiting_spacing_CAMB_lg)
        print("limiting_spacing_box= ",limiting_spacing_box_sm,"-",limiting_spacing_box_lg)
        print("limiting_spacing_surv=",limiting_spacing_surv)

        print("\nMINS AND MAXES:")
        print("CAMB:           ",self.ksph[0],"-",self.ksph[-1])
        print("box vec:         I jump straight to the r-grid so this doesn't live in a variable thus far")
        print("init par & perp: ",self.kpartrue_init[0],"-",self.kpartrue_init[-1],"&",self.kperptrue_init[0],"-",self.kperptrue_init[-1])
        print("surv par & perp: ",self.kpar[0],"-",self.kpar[-1],"&",self.kperp[0],"-",self.kperp[-1])
        
        # interp a sph binned CAMB/etc. MPS to get values over a cylindrically binned k-grid of interest
        k,Psph=self.get_mps(self.kmin_surv,self.kmax_surv)
        Psph=Psph.reshape((Psph.shape[1],))
        kpargrid,kperpgrid=np.meshgrid(self.kpar_surv,self.kperp_surv,indexing="ij")
        Pcyl=np.zeros((self.Nkpar_surv,self.Nkperp_surv))
        for i,kpar_val in enumerate(self.kpar_surv):
            for j,kperp_val in enumerate(self.kperp_surv):
                k_of_interest=np.sqrt(kpar_val**2+kperp_val**2)
                idx_closest_k=np.argmin(np.abs(k-k_of_interest)) # k-scalar in the CAMB MPS closest to the k-magnitude indicated by the kpar-kperp combination for that point in cylindrically binned Fourier space
                if (idx_closest_k==0): # start of array
                    idx_2nd_closest_k=1 # use hi
                elif (idx_closest_k==self.n_sph_modes-1): # end of array
                    idx_2nd_closest_k=self.n_sph_modes-2 # use lo
                else: # middle of array -> check if hi or lo is closer
                    k_neighb_lo=k[idx_closest_k-1]
                    k_neighb_hi=k[idx_closest_k+1]
                    if (np.abs(k_neighb_lo-k_of_interest)<np.abs(k_neighb_hi-k_of_interest)): # use k_neighb_lo
                        idx_2nd_closest_k=idx_closest_k-1
                    else:
                        idx_2nd_closest_k=idx_closest_k+1
                k_closest=k[idx_closest_k]
                k_2nd_closest=k[idx_2nd_closest_k]
                interp_slope=(Psph[idx_2nd_closest_k]-Psph[idx_closest_k])/(k_2nd_closest-k_closest)
                Pcyl[i,j]=interp_slope*(k_of_interest-k_closest)
        self.kpargrid_surv=kpargrid
        self.kperpgrid_surv=kperpgrid
        self.Pcyl=Pcyl

    def get_mps(self,minkh=1e-4,maxkh=1):
        """
        get matter power spectrum from CAMB

        args
        minkh = min value of k/h at which to calculate the MPS
        maxkh = max value of k/h at which to calculate the MPS 
        """
        z=[z]
        H0=pars_set_cosmo[0]
        h=H0/100.
        ombh2=pars_set_cosmo[1]
        omch2=pars_set_cosmo[2]
        As=pars_set_cosmo[3]*scale
        ns=pars_set_cosmo[4]

        pars_set_cosmo=camb.set_params(H0=H0, ombh2=ombh2, omch2=omch2, ns=ns, mnu=0.06,omk=0)
        pars_set_cosmo.InitPower.set_params(As=As,ns=ns,r=0)
        pars_set_cosmo.set_matter_power(redshifts=z, kmax=maxkh*h)
        results = camb.get_results(pars_set_cosmo)
        pars_set_cosmo.NonLinear = model.NonLinear_none
        kh,z,pk=results.get_matter_power_spectrum(minkh=minkh,maxkh=maxkh,npoints=self.n_sph_modes)
        return kh,pk

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
        if (self.Pcylshape!=self.Wcontshape):
            if(self.Pcylshape.T!=self.Wcontshape):
                assert(1==0), "window and power spec shapes must match"
            self.Wcont=self.Wcont.T # force P and Wcont to have the same shapes
        s0,s1=self.Pcylshape # by now, P and Wcont have the same shapes
        pad0lo,pad0hi=self.get_padding(s0)
        pad1lo,pad1hi=self.get_padding(s1)
        Wcontp=np.pad(self.Wcont,((pad0lo,pad0hi),(pad1lo,pad1hi)),"edge")
        self.Pcont_cyl=convolve(Wcontp,self.Pcyl,mode="valid") ### same update as calc_Pcont_asym
    
    def W_cyl_binned(self,beam_pars_to_use):
        """
        wrapper to multiply the LoS and flat sky approximation sky plane terms of the cylindrically binned window function, for the grid described by the k-parallel and k-perp modes of the survey of interest
        """
        self.sigLoS,self.fwhm_x,self.fwhm_y,self.r0
        sigLoS_use,fwhm_x_use,_,r0_use=beam_pars_to_use
        par_vec=np.exp(-self.deltakpar_surv**2*sigLoS_use**2)
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
        self.Wth=self.W_cyl_binned(self.perturbed_primary_beam_args)
        self.Wcont=self.Wtr-self.Wth
        self.Wcontshape=self.Wcont.shape
    
    def NvoxPracticalityWarning(Nvox,threshold_lo=75,threshold_hi=200):
        prefix="WARNING: the specified survey requires Nvox="
        if Nvox>threshold_hi:
            print(prefix+"{:4}, which may cause slow eval".format(Nvox))
        elif Nvox<threshold_lo:
            print(prefix+"{:4}, which is suspiciously coarse".format(Nvox))
        return None

    def calc_Pcont_asym(self,nkpar_box=15,nkperp_box=18,frac_tol=0.1):
        """
        calculate a cylindrically binned Pcont from an average over the power spectra formed from cylindrically-asymmetric-response-modulated brightness temp fields for a cosmological case of interest
        (you can still form a cylindrical summary statistic from brightness temp fields encoding effects beyond this symmetry)

        returns
        contaminant power, calculated as the difference of subtracted spectra with config space–multiplied "true" and "thought" instrument responses
        """

        tr=cosmo_stats(self.Lsurvbox,
                       P_fid=self.Ptruesph,Nvox=self.Nvoxbox,
                       primary_beam=Gaussian_primary,primary_beam_args=self.primary_beam_args,
                       Nk0=nkpar_box,Nk1=nkperp_box,
                       frac_tol=frac_tol,
                       k0bins_interp=self.kpar_surv,k1bins_interp=self.kperp_surv,
                       k_fid=self.ksph)
        th=cosmo_stats(self.Lsurvbox,
                       P_fid=self.Ptruesph,Nvox=self.Nvoxbox,
                       primary_beam=Gaussian_primary,primary_beam_args=self.perturbed_primary_beam_args,
                       Nk0=nkpar_box,Nk1=nkperp_box,
                       frac_tol=frac_tol,
                       k0bins_interp=self.kpar_surv,k1bins_interp=self.kperp_surv,
                       k_fid=self.ksph)
        
        tr.avg_realizations()
        th.avg_realizations()
        
        self.Ptrue=    tr.P_converged
        self.Pthought= th.P_converged
        self.Pcont_cyl=self.Ptrue_cyl-self.Pthought_cyl ### same update as calc_Pcont_asym