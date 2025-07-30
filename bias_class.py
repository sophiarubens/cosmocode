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

class window_calcs(object):
    def __init__(self,
                 kpar_surv,kperp_surv,                                  # set by survey properties
                 primary_beam_type,primary_beam_args,primary_beam_uncs, # primary beam considerations
                 pars_set_cosmo,pars_forecast,                          # eventually: build out the functionality for pars_forecast to differ nontrivially from pars_set_cosmo
                 z,n_sph_modes,dpar,                                    # conditioning the CAMB/etc. call
                 nu_ctr,delta_nu,                                       # for the survey of interest
                 evol_restriction_threshold=1./15.,                     # misc. numerical considerations
                 init_and_box_tol=0.05,CAMB_tol=0.05                    # 
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
        self.kpar_surv=kpar_surv
        self.Nkpar_surv=len(kpar_surv)
        self.kperp_surv=kperp_surv
        self.Nkperp_surv=len(kperp_surv)

        self.kmin_surv=np.sqrt(kpar_surv[ 0]**2+kperp_surv[ 0]**2)
        self.kmax_surv=np.sqrt(kpar_surv[-1]**2+kperp_surv[-1]**2)
        self.Lsurvbox= twopi/self.kmin
        self.Nvoxbox=  int(self.Lsurvbox*self.kmax/pi)
        self.NvoxPracticalityWarning(self.Nvoxbox)

        if (primary_beam_type.lower()=="gaussian"):
            self.sigLoS,self.fwhm_x,self.fwhm_y,self.r0=self.primary_beam_args
            self.epsLoS,self.epsx,self.epsy
        else:
            raise NotYetImplementedError
        self.primary_beam_type=primary_beam_type
        self.primary_beam_args=primary_beam_args
        self.primary_beam_uncs=primary_beam_uncs
        
        self.pars_set_cosmo=pars_set_cosmo
        self.pars_forecast=pars_forecast
        self.z=z
        self.n_sph_modes=n_sph_modes
        self.dpar=dpar
        self.nu_ctr=nu_ctr
        self.Deltanu=delta_nu
        self.bw=nu_ctr*evol_restriction_threshold # previously called NDeltanu
        self.Nchan=int(self.bw/self.Deltanu)

        self.kpargrid,self.kperpgrid,self.Pcyl=self.unbin_to_Pcyl(self)
        self.Pcylshape=self.Pcyl.shape

        ##
        limiting_spacing_surv=np.min([ self.kpar[1]-self.kpar[0], self.kperp[1]-self.kperp[0]])
        kmin_box_and_init=(1-init_and_box_tol)*self.kmin_surv
        kmax_box_and_init=(1+init_and_box_tol)*self.kmax_surv
        kmin_CAMB=(1-CAMB_tol)*kmin_box_and_init
        kmax_CAMB=(1+CAMB_tol)*kmax_box_and_init
        ksph,Ptruesph=get_mps(pars_set_cosmo,z,minkh=kmin_CAMB,maxkh=kmax_CAMB,n_sph_modes=n_sph_modes) # TESTING WHAT HAPPENS WHEN I DON'T DIVIDE BY h
        limiting_spacing_CAMB_sm=ksph[1]-ksph[0]
        limiting_spacing_CAMB_lg=ksph[-1]-ksph[-2]
        limiting_spacing_box_sm=2./(self.Nvoxbox*self.Lsurvbox*np.sqrt(3)*((np.sqrt(3)/2)-(1./self.Nvoxbox)))
        limiting_spacing_box_lg=self.Nvoxbox/(2.*self.Lsurvbox)
        Deltabox=self.Lsurvbox/self.Nvoxbox
        sky_plane_sigmas=self.r0*np.array([self.beamfwhm_x,self.beamfwhm_y])/np.sqrt(2*np.log(2))
        all_sigmas=np.concatenate((sky_plane_sigmas,[self.sigLoS]))
        if (np.any(all_sigmas<Deltabox)):
            raise NumericalDeltaError
        Nkpar_surv=len(kpar)
        Nkperp_surv=len(kperp)
        ##

    def get_padding(self,n):
            padding=n-1
            padding_lo=int(np.ceil(padding / 2))
            padding_hi=padding-padding_lo
            return padding_lo,padding_hi
    
    def calc_Pcont_cyl(self):
        self.calc_Wcont()
        if (self.Pcylshape!=self.Wcontshape):
            if(self.Pcylshape.T!=self.Wcontshape):
                assert(1==0), "window and power spec shapes must match"
            self.Wcont=self.Wcont.T # force P and Wcont to have the same shapes
        s0,s1=self.Pcylshape # by now, P and Wcont have the same shapes
        pad0lo,pad0hi=self.get_padding(s0)
        pad1lo,pad1hi=self.get_padding(s1)
        Wcontp=np.pad(self.Wcont,((pad0lo,pad0hi),(pad1lo,pad1hi)),"edge")
        self.Pcont_cyl=convolve(Wcontp,self.Pcyl,mode="valid")
    
    def calc_Wcont(self):
        """
        calculate the "contaminant" windowing amplitude that will help give rise to the so-called "contaminant power"
        (ignores fwhmy and epsfwhmy because of the limits of analytical cylindrical math, although they need to be passed when initializing the class object to avoid unpacking errors)
        """
        deltakpar=  self.kpar  - self.kpar[ self.Nkpar// 2] 
        deltakperp= self.kperp - self.kperp[self.Nkperp//2]
        par_vec_tr=  np.exp(-deltakpar**2* self.sigLoS                 **2)
        perp_vec_tr= np.exp(-(self.r0*self.fwhmx*                  deltakperp)**2/(2.*ln2))
        par_vec_th=  np.exp(-deltakpar**2*(self.sigLoS*(1-self.epsLoS))**2)
        perp_vec_th= np.exp(-(self.r0*self.fwhmx*(1-self.epsfwhmx)*deltakperp)**2/(2.*ln2))

        self.Wtr=self.stitch_normalize(par_vec_tr,perp_vec_tr)
        self.Wth=self.stitch_normalize(par_vec_th,perp_vec_th)
        self.Wcont=self.Wtr-self.Wth
        self.Wcontshape=self.Wcont.shape
    
    def stitch_normalize(vec1,vec2):
        arr1,arr2=np.meshgrid(vec1,vec2,indexing="ij")
        meshed=arr1*arr2
        rawsum=np.sum(meshed)
        if (rawsum!=0):
            return meshed/rawsum
        else:
            return meshed
    
    def NvoxPracticalityWarning(Nvox,threshold_lo=75,threshold_hi=200):
        prefix="WARNING: the specified survey requires Nvox="
        if Nvox>threshold_hi:
            print(prefix+"{:4}, which may cause slow eval".format(Nvox))
        elif Nvox<threshold_lo:
            print(prefix+"{:4}, which is suspiciously coarse".format(Nvox))
        return None

    def calc_Pcont_asym(self,nkpar_box=15,nkperp_box=18,n_realiz=5,init_and_box_tol=0.05,CAMB_tol=0.05):
        """
        calculate a cylindrically binned Pcont from an average over the power spectra formed from cylindrically-asymmetric-response-modulated brightness temp fields for a cosmological case of interest
        (you can still form a cylindrical summary statistic from brightness temp fields encoding effects beyond this symmetry)

        returns
        contaminant power, calculated as the difference of subtracted spectra with config space–multiplied "true" and "thought" instrument responses
        """
        
        Ptrue_realizations=   np.zeros((Nkpar_surv,Nkperp_surv,n_realiz))
        Pthought_realizations=np.zeros((Nkpar_surv,Nkperp_surv,n_realiz))
        bundled_args=(sigLoS,beamfwhm_x,beamfwhm_y,r0,)
        for i in range(n_realiz):
            t2=time.time()
            _,Tbox,rmags=generate_box(Ptruesph,ksph,Lsurvbox,Nvoxbox,custom_estimator2=custom_response,custom_estimator_args=bundled_args) #generate_box(P,k,Lsurvey,Nvox,custom_estimator2=False,custom_estimator_args=None) 
            t3=time.time()
            # print(">> Pcont calc: generated box from pspec - realization",i,t3-t2)
            if (i==0):
                t4=time.time()
                X,Y,Z=np.meshgrid(rmags,rmags,rmags,indexing="ij")
                response_true=    custom_response(X,Y,Z, sigLoS,           beamfwhm_x,          beamfwhm_y,          r0)
                response_thought= custom_response(X,Y,Z, sigLoS*(1-epsLoS),beamfwhm_x*(1-eps_x),beamfwhm_y*(1-eps_y),r0)
                t5=time.time()
                # print(">> Pcont calc: generated responses",t5-t4)
            t6=time.time()
            T_x_true_resp=   Tbox* response_true
            T_x_thought_resp=Tbox* response_thought
            t7=time.time()
            # print(">> Pcont calc: multiplied box and instrument response - realization",i,t7-t6)
            ktrue_init,    Ptrue_init=    generate_P(T_x_true_resp,    "lin",Lsurvbox,nkpar_box,Nk1=nkperp_box, custom_estimator2=custom_response2,custom_estimator_args=bundled_args) # WAS LIN BUT NUMERICS WERE BAD
            kpartrue_init,kperptrue_init=ktrue_init
            limiting_spacing_init=np.min([kpartrue_init[1]-kpartrue_init[0],kperptrue_init[1]-kperptrue_init[0]])
            t8=time.time()
            kthought_init, Pthought_init= generate_P(T_x_thought_resp, "lin",Lsurvbox,nkpar_box,Nk1=nkperp_box, custom_estimator2=custom_response2,custom_estimator_args=bundled_args)
            k_survey=(kpar,kperp)
            t9=time.time()
            # print(">> Pcont calc: generated pspecs from modulated boxes - realization",i,t9-t7)
            _,   Ptrue= interpolate_P(Ptrue_init,    ktrue_init,    k_survey, avoid_extrapolation=False) # the returned k are the same as the k-modes passed in k_survey
            _,Pthought= interpolate_P(Pthought_init, kthought_init, k_survey, avoid_extrapolation=False)
            t10=time.time()
            # print(">> Pcont calc: re-binned pspecs to k-modes of interest - realization",i,t10-t9)
            Ptrue_realizations[:,:,i]=    Ptrue
            Pthought_realizations[:,:,i]= Pthought
            t11=time.time()
            print(">> Pcont calc: generated modulated power spectra - realization",i,t11-t2)
            if i==0:
                print("END OF ITERATION 0 - COMPARISONS")
                print("\nLIMITING SPACINGS")
                print("limiting_spacing_CAMB=",limiting_spacing_CAMB_sm,"-",limiting_spacing_CAMB_lg)
                print("limiting_spacing_box= ",limiting_spacing_box_sm,"-",limiting_spacing_box_lg)
                print("limiting_spacing_init=",limiting_spacing_init)
                print("limiting_spacing_surv=",limiting_spacing_surv)

                print("\nMINS AND MAXES:")
                print("CAMB:           ",ksph[0],"-",ksph[-1])
                print("box vec:         I jump straight to the r-grid so this doesn't live in a variable thus far")
                print("init par & perp: ",kpartrue_init[0],"-",kpartrue_init[-1],"&",kperptrue_init[0],"-",kperptrue_init[-1])
                print("surv par & perp: ",kpar[0],"-",kpar[-1],"&",kperp[0],"-",kperp[-1])
        Ptrue=    np.mean(Ptrue_realizations,    axis=-1)
        Pthought= np.mean(Pthought_realizations, axis=-1)
        Pthought=0 # FOR DIAGNOSTIC PURPOSES WHILE AVOIDING RESTRUCTURING MY CODE: CALL IT PCONT BUT REALLY HAVE IT REFLECT PTRUE
        t12=time.time()
        print(">> Pcont calc: averaged over statistical realizations to obtain Ptrue and Pthought",t12-t11)
        Pcont=Ptrue-Pthought
        t13=time.time()
        print(">> Pcont calc: subtracted Pthought from Ptrue",t13-t12)
        return Pcont