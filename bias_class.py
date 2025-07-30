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

ln2=np.log(2.)

class NotYetImplementedError(Exception):
    pass
class NvoxPracticalityError(Exception):
    pass
class NumericalDeltaError(Exception):
    pass

class window_calcs(object):
    def __init__(self,
                 kpar,kperp,                                            # set by survey properties
                 primary_beam_type,primary_beam_args,primary_beam_uncs, # primary beam considerations
                 pars_set_cosmo,pars_forecast,                          # eventually: build out the functionality for pars_forecast to differ nontrivially from pars_set_cosmo
                 z,n_sph_modes,dpar,                                    # conditioning the CAMB/etc. call
                 nu_ctr,delta_nu,                                       # for the survey of interest
                 evol_restriction_threshold=1./15.                      # misc. numerical considerations
                 ):
        """
        kpar,kperp                 :: (Nkpar,),(Nkperp,) of floats :: mono incr. cyl binned curvey modes       :: 1/Mpc
        primary_beam               :: callable                     :: power beam in Cartesian coords           :: ---
        primary_beam_type          :: str                          :: implement soon: Airy etc.                :: ---
        primary_beam_args          :: (N_args,) of floats          :: Gaussian: "μ"s and "σ"s                  :: Gaussian: sigLoS, r0 in Mpc; fwhm_x, fwhm_y in rad
        primary_beam_uncs          :: (N_uncertain_args) of floats :: Gaussian: frac uncs epsLoS, epsfwhm{x/y} :: ---
        pars_set_cosmo             :: (N_fid_pars,) of floats      :: params to condition a CAMB/etc. call     :: as found in ΛCDM
        pars_forecast              :: (N_forecast_pars,) of floats :: params for which you'd like to forecast  :: as found in ΛCDM
        z                          :: float                        :: z of fiducial MPS for CAMB/etc. call     :: ---
        n_sph_modes                :: int                          :: # modes to put in CAMB/etc. MPS          :: ---
        dpar                       :: (N_forecast_pars,) of floats :: initial guess of num. dif. step sizes    :: same as for pars_forecast
        nu_ctr                     :: float                        :: central freq for survey of interest      :: MHz
        delta_nu                   :: float                        :: channel width for survey of interest     :: MHz
        evol_restriction_threshold :: float                        :: ~$\frac{\Delta z}{z}$ w/in survey box    :: ---
        """
        self.kpar=kpar
        self.Nkpar=len(kpar)
        self.kperp=kperp
        self.Nkperp=len(kperp)

        if (primary_beam_type.lower()=="gaussian"):
            self.sigLoS,self.fwhm_x,self.fwhm_y,self.r0=self.primary_beam_args
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

    def get_padding(self,n):
            padding=n-1
            padding_lo=int(np.ceil(padding / 2))
            padding_hi=padding-padding_lo
            return padding_lo,padding_hi
    
    def higher_dim_conv(self,P,Wcont):
        Pshape=P.shape
        Wcontshape=Wcont.shape
        if (Pshape!=Wcontshape):
            if(Pshape.T!=Wcontshape):
                assert(1==0), "window and pspec shapes must match"
            Wcont=Wcont.T # force P and Wcont to have the same shapes
        s0,s1=Pshape # by now, P and Wcont have the same shapes
        pad0lo,pad0hi=self.get_padding(s0)
        pad1lo,pad1hi=self.get_padding(s1)
        Wcontp=np.pad(Wcont,((pad0lo,pad0hi),(pad1lo,pad1hi)),"edge")
        conv=convolve(Wcontp,P,mode="valid")
        return conv
    
    def W_cyl_binned(self):
        """
        wrapper to multiply the LoS and flat sky approximation sky plane terms of the cylindrically binned window function, for the grid described by the k-parallel and k-perp modes of the survey of interest
        """
        deltakpar=self.kpar-self.kpar[self.Nkpar//2] 
        deltakperp=self.kperp-self.kperp[self.Nkperp//2]
        par_vec= np.exp(-deltakpar**2*self.sigLoS**2)
        perp_vec=np.exp(-(self.r0*self.fwhmbeam*deltakperp)**2/(2.*ln2))
        par_arr,perp_arr=np.meshgrid(par_vec,perp_vec,indexing="ij")
        meshed=par_arr*perp_arr # I really do want elementwise multiplication
        rawsum=np.sum(meshed)
        if (rawsum!=0): # normalize, but protect against division-by-zero errors
            normed=meshed/rawsum
        else:
            normed=meshed
        return normed
    
    def calc_Wcont(self):
        Wtrue=self.W_cyl_binned()
    ############################################################################################ THINGS BELOW THIS LINE HAVE BEEN LIFTED STRAIGHT FROM bias_helper_fcns.py AND HAVE NOT YET BEEN MODIFIED

    def calc_Wcont(kpar,kperp,sigLoS,r0,fwhmbeam,epsLoS,epsbeam): 
        """
        calculate the "contaminant" windowing amplitude that will help give rise to the so-called "contaminant power"
        """
        Wtrue=   W_cyl_binned(kpar,kperp,sigLoS,           r0,fwhmbeam            )
        Wthought=W_cyl_binned(kpar,kperp,sigLoS*(1-epsLoS),r0,fwhmbeam*(1-epsbeam)) # FOR NOW: BAKED IN THAT THE "THOUGHT" WIDTH RESPONSE PARAMS ARE UNDERESTIMATES FOR POSITIVE EPS
        Wthought=0 # FOR DIAGNOSTIC PURPOSES WHILE AVOIDING RESTRUCTURING MY CODE: CALL IT PCONT BUT REALLY HAVE IT REFLECT PTRUE
        return Wtrue-Wthought

    def calc_Pcont_cyl(kpar,kperp,sigLoS,r0,fwhmbeam,pars_set_cosmo,epsLoS,epsbeam,z,n_sph_modes): 
        """
        calculate the cylindrically binned "contaminant power," following from the true and perceived window functions
        """
        Wcont=calc_Wcont(kpar,kperp,sigLoS,r0,fwhmbeam,epsLoS,epsbeam)
        print(">> Pcont calc: calculated Wcont")
        kpargrid,kperpgrid,P=unbin_to_Pcyl(kpar,kperp,z,pars_set_cosmo=pars_set_cosmo,n_sph_modes=n_sph_modes)
        print(">> Pcont calc: unbinned CAMB pspec to cyl")
        ###
        np.save("cyl_Wcont.npy",Wcont)
        np.save("cyl_P.npy",P)
        np.save("cyl_kpargrid.npy",kpargrid)
        np.save("cyl_kperpgrid.npy",kperpgrid)
        ###
        Pcont=higher_dim_conv(P,Wcont) # new prototype, not symmetric under exchange of args: higher_dim_conv(P,Wcont)
        print(">> Pcont calc: convolved P and Wcont")
        return Pcont

    def NvoxPracticalityWarning(Nvox,threshold_lo=75,threshold_hi=200):
        prefix="WARNING: the specified survey requires Nvox="
        if Nvox>threshold_hi:
            print(prefix+"{:4}, which may cause slow eval".format(Nvox))
        elif Nvox<threshold_lo:
            print(prefix+"{:4}, which is suspiciously coarse".format(Nvox))
        return None 

    def get_L_N(kmin,kmax):
        Lsurvbox=twopi/kmin
        Nvoxbox=int(Lsurvbox*kmax/pi)
        if (Nvoxbox>200):
            NvoxPracticalityWarning(Nvoxbox)
        return Lsurvbox,Nvoxbox

    def calc_Pcont_asym(pars_set_cosmo,z,kpar,kperp,sigLoS,epsLoS,r0,beamfwhm_x,beamfwhm_y,eps_x,eps_y,Nvox=150,n_sph_modes=500,nkpar_box=15,nkperp_box=18,n_realiz=5,init_and_box_tol=0.05,CAMB_tol=0.05):
        """
        calculate a cylindrically binned Pcont from an average over the power spectra formed from cylindrically-asymmetric-response-modulated brightness temp fields for a cosmological case of interest
        (you can still form a cylindrical summary statistic from brightness temp fields encoding effects beyond this symmetry)

        args
        beamfwhm_x = FWHM of the beam in one       polarization direction
        beamfwhm_y = "                 " the other "                    "
        eps_x      = fractional uncertainty in beamfwhm_x
        eps_y      = fractional uncertainty in beamfwhm_y
        Nvox   = number of voxels per side to use when constructing random realization Tb cubes

        returns
        contaminant power, calculated as the difference of subtracted spectra with config space–multiplied "true" and "thought" instrument responses
        """
        t0=time.time()
        h=pars_set_cosmo[0]/100 # typical disclaimer about cosmo param order being baked in...
        kmin_surv=np.min([kpar[0], kperp[0]] )
        kmax_surv=np.max([kpar[-1],kperp[-1]])
        limiting_spacing_surv=np.min([ kpar[1]-kpar[0], kperp[1]-kperp[0]])
        kmin_box_and_init=(1-init_and_box_tol)*kmin_surv
        kmax_box_and_init=(1+init_and_box_tol)*kmax_surv
        kmin_CAMB=(1-CAMB_tol)*kmin_box_and_init
        kmax_CAMB=(1+CAMB_tol)*kmax_box_and_init
        ksph,Ptruesph=get_mps(pars_set_cosmo,z,minkh=kmin_CAMB,maxkh=kmax_CAMB,n_sph_modes=n_sph_modes) # TESTING WHAT HAPPENS WHEN I DON'T DIVIDE BY h
        limiting_spacing_CAMB_sm=ksph[1]-ksph[0]
        limiting_spacing_CAMB_lg=ksph[-1]-ksph[-2]
        np.save("ksph_for_asym.npy",ksph)
        np.save("Ptruesph_for_asym.npy",Ptruesph)
        t1=time.time()
        print(">> Pcont calc: sourced pspec from CAMB",t1-t0)
        # Lsurvbox,Nvoxbox=get_L_N_for_box(kpar,kperp)
        Lsurvbox,Nvoxbox=get_L_N(kmin_box_and_init,kmax_box_and_init)
        print("Lsurvbox,Nvoxbox=",Lsurvbox,Nvoxbox)
        limiting_spacing_box_sm=2./(Nvoxbox*Lsurvbox*np.sqrt(3)*((np.sqrt(3)/2)-(1./Nvoxbox)))
        limiting_spacing_box_lg=Nvoxbox/(2.*Lsurvbox)
        Deltabox=Lsurvbox/Nvoxbox
        sky_plane_sigmas=r0*np.array([beamfwhm_x,beamfwhm_y])/np.sqrt(2*np.log(2))
        all_sigmas=np.concatenate((sky_plane_sigmas,[sigLoS]))
        if (np.any(all_sigmas<Deltabox)):
            raise NumericalDeltaError
        Nkpar_surv=len(kpar)
        Nkperp_surv=len(kperp)
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

    def custom_response(X,Y,Z,sigLoS,beamfwhm_x,beamfwhm_y,r0):
        """
        "custom" response function using the approximation where there is a Gaussian along the LoS and another in the sky plane

        args
        X,Y,Z      = meshgridded (indexing="ij") (Nvox,Nvox,Nvox) boxes 
        sigLoS     = characteristic width of the instrument response function along the line of sight
        beamfwhm_x = x-pol power beam fwhm
        beamfwhm_y = y-pol power beam fwhm
        r0         = central comoving distance of the survey volume 

        returns
        (Nvox,Nvox,Nvox) Cartesian box (z=LoS direction), centred at r0, sampling the response fcn at each point
        """
        response=np.exp(-(Z/(2*sigLoS))**2 -ln2*((X/beamfwhm_x)**2+(Y/beamfwhm_y)**2)/r0**2)
        return response

    def custom_response2(X,Y,Z,sigLoS,beamfwhm_x,beamfwhm_y,r0):
        """
        "custom" response function using the approximation where there is a Gaussian along the LoS and another in the sky plane

        args
        X,Y,Z      = meshgridded (indexing="ij") (Nvox,Nvox,Nvox) boxes 
        sigLoS     = characteristic width of the instrument response function along the line of sight
        beamfwhm_x = x-pol power beam fwhm
        beamfwhm_y = y-pol power beam fwhm
        r0         = central comoving distance of the survey volume 

        returns
        (Nvox,Nvox,Nvox) Cartesian box (z=LoS direction), centred at r0, sampling the response fcn at each point
        """
        
        return custom_response(X,Y,Z,sigLoS,beamfwhm_x,beamfwhm_y,r0)**2

    def unbin_to_Pcyl(kpar,kperp,z,pars_set_cosmo=pars_set_cosmo_Planck18,n_sph_modes=500):  
        """
        interpolate a spherically binned CAMB MPS to provide MPS values for a cylindrically binned k-grid of interest (nkpar x nkperp)
        """
        h=pars_set_cosmo[0]/100.
        kmin=np.sqrt(kpar[0]**2+kperp[0]**2)
        kmax=np.sqrt(kpar[-1]**2+kperp[-1]**2)
        k,Psph=get_mps(pars_set_cosmo,z,minkh=kmin/h,maxkh=kmax/h,n_sph_modes=n_sph_modes)
        Psph=Psph.reshape((Psph.shape[1],))
        kpargrid,kperpgrid=np.meshgrid(kpar,kperp,indexing="ij")
        Pcyl=np.zeros((len(kpar),len(kperp)))
        for i,kpar_val in enumerate(kpar):
            for j,kperp_val in enumerate(kperp):
                k_of_interest=np.sqrt(kpar_val**2+kperp_val**2)
                idx_closest_k=np.argmin(np.abs(k-k_of_interest)) # k-scalar in the CAMB MPS closest to the k-magnitude indicated by the kpar-kperp combination for that point in cylindrically binned Fourier space
                if (idx_closest_k==0): # start of array
                    idx_2nd_closest_k=1 # use hi
                elif (idx_closest_k==n_sph_modes-1): # end of array
                    idx_2nd_closest_k=n_sph_modes-2 # use lo
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
        return kpargrid,kperpgrid,Pcyl

    scale=1e-9
    def get_mps(pars_set_cosmo,z,minkh=1e-4,maxkh=1,n_sph_modes=500):
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
        # camb.dark_energy.DarkEnergyEqnOfState.set_params(w=, wa=)
        pars_set_cosmo.InitPower.set_params(As=As,ns=ns,r=0)
        pars_set_cosmo.set_matter_power(redshifts=z, kmax=maxkh*h)
        results = camb.get_results(pars_set_cosmo)
        pars_set_cosmo.NonLinear = model.NonLinear_none
        kh,z,pk=results.get_matter_power_spectrum(minkh=minkh,maxkh=maxkh,npoints=n_sph_modes)
        return kh,pk

    def cyl_partial(pars_set_cosmo,z,n,dpar,kpar,kperp,n_sph_modes=500,ftol=1e-6,eps=1e-16,maxiter=5):  
        """
        args
        n       = take the partial derivative WRT the nth parameter in p
        ftol    = fractional tolerance relating to the scale of the function (defined for points of interest)
        eps     = tiny offset factor to protect against numerical division-by-zero errors
        maxiter = maximum number of times to let the step size optimization attempt recurse before "giving up" and using the most recent guess

        returns
        cylindrically binned matter power spectrum partial WRT one cosmo parameter (nkpar x nkperp)
        """
        done=False
        iter=0
        dparn=dpar[n]
        pcopy=pars_set_cosmo.copy()
        pndispersed=pcopy[n]+np.linspace(-2,2,5)*dparn

        pcopy=pars_set_cosmo.copy()
        _,_,Pcyl_base=unbin_to_Pcyl(kpar,kperp,z,pars_set_cosmo=pcopy,n_sph_modes=n_sph_modes)
        P0=np.mean(np.abs(Pcyl_base))+eps
        tol=ftol*P0 # generalizes tol=ftol*f0 from 512

        pcopy[n]=pcopy[n]+2*dparn
        _,_,Pcyl_2plus=unbin_to_Pcyl(kpar,kperp,z,pars_set_cosmo=pcopy,n_sph_modes=n_sph_modes)
        pcopy=pars_set_cosmo.copy()
        pcopy[n]=pcopy[n]-2*dparn
        _,_,Pcyl_2minu=unbin_to_Pcyl(kpar,kperp,z,pars_set_cosmo=pcopy,n_sph_modes=n_sph_modes)
        deriv1=(Pcyl_2plus-Pcyl_2minu)/(4*dpar[n])

        pcopy[n]=pcopy[n]+dparn
        _,_,Pcyl_plus=unbin_to_Pcyl(kpar,kperp,z,pars_set_cosmo=pcopy,n_sph_modes=n_sph_modes)
        pcopy=pars_set_cosmo.copy()
        pcopy[n]=pcopy[n]-dparn
        _,_,Pcyl_minu=unbin_to_Pcyl(kpar,kperp,z,pars_set_cosmo=pcopy,n_sph_modes=n_sph_modes)
        deriv2=(Pcyl_plus-Pcyl_minu)/(2*dpar[n])

        while (done==False):
            if (np.any(Pcyl_plus-Pcyl_minu)<tol): # consider relaxing this to np.any if it ever seems like too strict a condition?!
                estimate=(4*deriv2-deriv1)/3
                return estimate # higher-order estimate
            else:
                pnmean=np.mean(np.abs(pndispersed)) # the np.abs part should be redundant because, by this point, all the k-mode values and their corresponding dpns and Ps should be nonnegative, but anyway... numerical stability or something idk
                Psecond=np.abs(2*Pcyl_base-Pcyl_minu-Pcyl_plus)/dpar[n]**2
                dparn=np.sqrt(eps*pnmean*P0/Psecond)
                iter+=1
                if iter==maxiter:
                    print("failed to converge in {:d} iterations".format(maxiter))
                    fallback=(4*deriv2-deriv1)/3
                    print("RETURNING fallback")
                    return fallback

    def build_cyl_partials(pars_set_cosmo,z,n_sph_modes,kpar,kperp,dpar):
        """
        builds a (npfore,nkpar,nkperp) array of the partials of the cylindrically binned MPS WRT each cosmo param in the forecast
        """
        nkpar=len(kpar)
        nkperp=len(kperp)
        nprm=len(pars_set_cosmo)
        V=np.zeros((nprm,nkpar,nkperp))
        for n in range(nprm):
            V[n,:,:]=cyl_partial(pars_set_cosmo,z,n,dpar,kpar,kperp,n_sph_modes=n_sph_modes)
        return V

    def bias(partials,unc, kpar,kperp,sigLoS,r0,fwhmbeam0,pars_set_cosmo,epsLoS,epsbeam0,z,n_sph_modes,savename=None,cyl_sym_resp=True, fwhmbeam1=1e-3,epsbeam1=0.1,Nvox=150,recalc_Pcont=False,n_realiz=5):
        """
        args
        partials = npfore x nkpar x nkperp array where each slice of constant 0th (nprm) index is an nkpar x nkperp array of the MPS's partial WRT a particular parameter in the forecast
        unc      = nkpar x nkperp array describing the standard deviations at each cylindrically binned k-mode

        returns
        (npfore,) vector of biases resulting from beam mismodelling for the parameters of interest in the forecast
        """
        V=0.0*partials # still want the same shape as the vector of partials, even though this is different than for the spherical case
        nprm=partials.shape[0]
        uncsh0,uncsh1=unc.shape
        partsh0,partsh1,partsh2=partials.shape
        if (uncsh0==partsh2 and uncsh1==partsh1):
            unc=unc.T

        for i in range(nprm):
            V[i,:,:]=partials[i,:,:]/unc # elementwise division for an nkpar x nkperp slice
        V_completely_transposed=np.transpose(V,axes=(2,1,0)) # from the docs: "For an n-D array, if axes are given, their order indicates how the axes are permuted"
        F=np.einsum("ijk,kjl->il",V,V_completely_transposed)
        print("computed F")
        if recalc_Pcont:
            if cyl_sym_resp:
                Pcont=calc_Pcont_cyl(kpar,kperp,sigLoS,r0,fwhmbeam0,pars_set_cosmo,epsLoS,epsbeam0,z,n_sph_modes)
            else:
                Pcont=calc_Pcont_asym(pars_set_cosmo,z,
                                    kpar,kperp,
                                    sigLoS,epsLoS,r0,fwhmbeam0,fwhmbeam1,epsbeam0,epsbeam1,
                                    Nvox=Nvox,n_sph_modes=n_sph_modes,n_realiz=n_realiz) 
        else:
            Pcont=np.load(savename)

        print("computed Pcont")
        np.save("Pcont_"+savename+".npy",Pcont)
        Pcont_div_sigma=Pcont/unc
        B=np.einsum("jk,ijk->i",Pcont_div_sigma,V)
        print("computed B")
        bias=(np.linalg.inv(F)@B).reshape((F.shape[0],))
        print("computed b")
        return bias

    def printparswbiases(pars_set_cosmo,parnames,biases):
        """
        args
        parnames = (npfore,) vector of strings: names of the parameters in the forecast (assumed to be in the same order as pars_set_cosmo)
        biases   = (npfore,) vector: biases in estimating the cosmo params in the forecast resulting from inadvertent beam mismodelling (in the same order as pars_set_cosmo and parnames)

        returns
        None (fcn prints a formatted summary of the forecasting pipeline calculation)
        """
        print("\n\nbias calculation results for the survey described above.................................")
        print("........................................................................................")
        for p,par in enumerate(pars_set_cosmo):
            print('{:12} = {:-10.3e} with bias {:-12.5e} (fraction = {:-10.3e})'.format(parnames[p], par, biases[p], biases[p]/par))
        return None