import numpy as np
from scipy.interpolate import interpn,interp1d
from numpy.fft import fftshift,ifftshift,fftn,irfftn,fftfreq

"""
this module helps connect ensemble-averaged power spectrum estimates 
and cosmological brighness temperature boxes for three main use cases:
1. generate a power spectrum that describes the statistics of a cosmo box
2. generate realizations of a cosmo box consistent with a known power spectrum
3. interpolate a power spectrum
"""

pi=np.pi
twopi=2.*pi
maxfloat= np.finfo(np.float64).max
maxint=   np.iinfo(np.int64  ).max

class ResolutionError(Exception):
    pass
class NotYetImplementedError(Exception):
    pass
class UnsupportedBinningMode(Exception):
    pass
class UnsupportedPrimaryBeamType(Exception):
    pass
class NotEnoughInfoError(Exception):
    pass
class ConflictingInfoError(Exception): # make these inherit from each other (or something) to avoid repetitive code
    pass
def extrapolation_warning(regime,want,have):
    print("WARNING: if extrapolation is permitted in the interpolate_P call, it will be conducted for {:15s} (want {:9.4}, have{:9.4})".format(regime,want,have))
    return None

class cosmo_stats(object):
    def __init__(self,Lsurvey,T=None,P=None,Nvox=None,primary_beam=None,primary_beam_args=None,primary_beam_type="Gaussian"):
        """
        Lsurvey           = float                      :: side length of cosmo box       :: Mpc
        T                 = (Nvox,Nvox,Nvox) floats    :: cosmo box                      :: K
        P                 = if Nk1=0: (Nk0,) floats    :: sph... binned power spectrum   :: K^2 Mpc^3
                            if Nk1>0: (Nk0,Nk1) floats :: cyl... "                     " ::
        Nvox              = float                      :: # voxels PER SIDE of cosmo box :: ---
        primary_beam      = callable                   :: power beam in Cartesian coords :: ---
        primary_beam_args = tuple of floats            :: Gaussian: "μ"s and "σ"s        :: Gaussian: sigLoS, r0 in Mpc; fwhm_x, fwhm_y in rad
        primary_beam_type = string                     :: implement later: Airy etc.     :: ---
        """
        self.Lsurvey=Lsurvey

        # spectrum and box
        if ((T==None) and ((Nvox==None) or (P==None))):
            raise NotEnoughInfoError
        else:
            self.T=T
            if ((Nvox is not None) and (Nvox!=T.shape[0])):
                raise ConflictingInfoError
            else:
                self.Nvox=self.T.shape[0]
            
            if (P is not None):
                Pshape=P.shape
                Pdims=len(Pshape)
                if (Pdims==2):
                    self.Nk0,self.Nk1=Pshape # interpret as kpar,kperp, but use index-y nomenclature
                elif (Pdims==1):
                    self.Nk0=Pshape
                    self.Nk1=0
                else:
                    raise UnsupportedBinningMode

        # primary beam
        self.primary_beam=primary_beam
        self.primary_beam_args=primary_beam_args
        self.primary_beam_type=primary_beam_type
        if (self.primary_beam is not None):
            if self.primary_beam_type=="Gaussian":
                self.sigLoS,self.fwhm_x,self.fwhm_y,self.r0=self.custom_estimator_args
                self.evaled_response=self.primary_beam(self.xx_grid,self.yy_ygrid,self.zz_zgrid,self.sigLoS,self.fwhm_x,self.fwhm_y,self.r0)
            else:
                UnsupportedPrimaryBeamType
            self.Veff=np.sum(self.evaled_response*self.d3r) # rectangular sum method
            self.evaled_response[self.evaled_response==0]=maxfloat # protect against division-by-zero errors
        else: # identity primary beam
            self.Veff=self.Lsurvey**3
            self.evaled_response=np.ones((self.Nvox,self.Nvox,self.Nvox))
        
        # config space
        self.Delta=self.Lsurvey/self.Nvox                            # voxel side length
        self.d3r=self.Delta**3                                       # volume element / voxel volume
        self.r_vec_for_box=self.Lsurvey*fftshift(fftfreq(self.Nvox))           # one Cartesian coordinate axis
        self.xx_grid,self.yy_grid,self.zz_grid=np.meshgrid(self.r_vec_for_box,
                                                           self.r_vec_for_box,
                                                           self.r_vec_for_box,
                                                           indexing="ij")      # box-shaped Cartesian coords
        self.r_grid=np.sqrt(self.xx_grid**2+self.yy_grid**2+self.zz_grid**2)   # r magnitudes at each voxel

        # Fourier space
        self.Deltak=twopi/self.Lsurvey                                     # voxel side length
        self.d3k=self.Deltak**3                                            # volume element / voxel volume
        self.k_vec_for_box=twopi*fftshift(fftfreq(self.Nvox,d=self.Delta))     # one Cartesian coordinate axis
        self.kx_grid,self.ky_grid,self.kz_grid=np.meshgrid(self.k_vec_for_box,
                                                           self.k_vec_for_box,
                                                           self.k_vec_for_box,
                                                           indexing="ij")      # box-shaped Cartesian coords   
        
            # sph binning
        self.k_grid=      np.sqrt(self.kx_grid**2+self.ky_grid**2+self.kz_grid**2)   # k magnitudes for each voxel
        self.sph_bin_indices=      np.digitize(self.k_grid,self.k0bins)              # sph bin that each voxel falls into
        self.sph_bin_indices_1d=   np.reshape(self.sph_bin_indices, (self.Nvox**3,)) # 1d version of ^ (compatible with np.bincount)

            # cyl binning
        self.kpar_column= np.abs(self.k_vec_for_box)                                          # magnitudes of kpar for a representative column along the line of sight (z-like)
        self.kperp_slice= np.sqrt(self.kx_grid**2+self.ky_grid**2)[:,:,0]                     # magnitudes of kperp for a representative slice transverse to the line of sight (x- and y-like)
        self.perpbin_indices_slice=    np.digitize(self.kperp_slice,self.k1bins)              # cyl kperp bin that each voxel falls into
        self.perpbin_indices_slice_1d= np.reshape(self.perpbin_indices_slice,(self.Nvox**2,)) # 1d version of ^ (compatible with np.bincount)
        self.parbin_indices_column=    np.digitize(self.kpar_column,self.k0bins)              # cyl kpar bin that each voxel falls into

        # placeholders (pre-condition the warnings)
        self.k0bins=None
        self.k1bins=None
        self.k0bins_interp=None
        self.k1bins_interp=None
        self.limiting_spacing_0=None
        self.limiting_spacing_1=None
        self.P_interp=None

    def calc_bins(self,Nki):
        kmax=pi/self.Delta # no factor of two = handle the k-coordinate +/- in the box (limits the upper bound on which coordinate magnitudes the box can tell you about)
        kmin=twopi/self.Lsurvey

        if (self.binning_mode=="log"):
            kbins=np.logspace(np.log10(kmin),np.log10(kmax),num=Nki)
            limiting_spacing=twopi*(10.**(kmax)-10.**(kmax-(np.log10(self.Nvox)/Nki)))
        elif (self.binning_mode=="lin"):
            kbins=np.linspace(kmin,kmax,Nki)
            limiting_spacing=twopi*(0.5*self.Nvox-1)/(Nki) # version for a kmax that is "aware that" there are +/- k-coordinates in the box
        else:
            raise UnsupportedBinningMode
        return kbins,limiting_spacing # kbins            -> floors of the bins to which the power spectrum will be binned (along one axis)
                                      # limiting_spacing -> smallest spacing between adjacent bins (uniform if linear; otherwise, depends on the binning strategy)
    
    def establish_bins(self):
        self.k0bins,self.limiting_spacing_0=self.calc_bins(self.Nk0)
        if self.limiting_spacing_0<self.deltak_box: # trying to bin more finely than the box can tell you about (guaranteed to have >=1 empty bin)
            raise ResolutionError
        
        if (self.Nk1>0):
            self.k1bins,self.limiting_spacing_1=self.calc_bins(self.Nk1)
            if (self.limiting_spacing_1<self.deltak_box): # idem ^
                raise ResolutionError
        else:
            self.k1bins=None
            
    def generate_P(self,Nk0,Nk1=0,binning_mode="lin"): # need to make sure this is only called when 
        if (Nk0!=self.Nk0):
            print("WARNING: overwriting Nk0 from init:"+str(self.Nk0)+" -> "+str(Nk0)+" (bins, P will also change)")
        if (Nk1!=self.Nk1):
            print("WARNING: overwriting Nk1 from init:"+str(self.Nk1)+" -> "+str(Nk1)+" (bins, P will also change)")
        if (self.T==None):
            raise NotEnoughInfoError # can't calculate a power spectrum if no box is present
        if (self.P is not None):
            print("WARNING: overwriting P")
        
        self.Nk0=Nk0
        self.Nk1=Nk1
        self.binning_mode=binning_mode
        self.establish_bins()

        T_no_primary_beam=  self.T/self.evaled_response
        T_tilde=            fftshift(fftn((ifftshift(T_no_primary_beam)*self.d3r)))
        modsq_T_tilde=     (T_tilde*np.conjugate(T_tilde)).real

        if (self.Nk1==0): # bin to sph
            modsq_T_tilde_1d= np.reshape(modsq_T_tilde,    (self.Nvox**3,))

            sum_modsq_T_tilde= np.bincount(self.sph_bin_indices_1d, 
                                           weights=modsq_T_tilde_1d, 
                                           minlength=self.Nk0)       # for the ensemble avg: sum    of modsq_T_tilde values in each bin
            N_modsq_T_tilde=   np.bincount(self.sph_bin_indices_1d,
                                           minlength=self.Nk0)       # for the ensemble avg: number of modsq_T_tilde values in each bin
            sum_modsq_T_tilde_truncated=sum_modsq_T_tilde[:-1]
            N_modsq_T_tilde_truncated=  N_modsq_T_tilde[:-1]
        else:             # bin to cyl
            sum_modsq_T_tilde= np.zeros((self.Nk0+1,self.Nk1+1)) # for the ensemble avg: sum    of modsq_T_tilde values in each bin  ...upon each access, update the kparBIN row of interest, but all Nkperp columns
            N_modsq_T_tilde=   np.zeros((self.Nk0+1,self.Nk1+1)) # for the ensemble avg: number of modsq_T_tilde values in each bin
            for i in range(self.Nvox): # iterate over the kpar axis of the box to capture all LoS slices
                if (i==0): # stats for the representative "bull's eye" slice transverse to the LoS
                    slice_bin_counts=np.bincount(self.perpbin_indices_slice_1d, minlength=self.Nk1)
                modsq_T_tilde_slice= modsq_T_tilde[:,:,i]                    # take the slice of interest of the preprocessed box values !! still treating kpar as z-like
                modsq_T_tilde_slice_1d= np.reshape(modsq_T_tilde_slice, 
                                                   (self.Nvox**2,))          # reshape to 1D for bincount compatibility
                current_binsums= np.bincount(self.perpbin_indices_slice_1d,
                                             weights=modsq_T_tilde_slice_1d, 
                                             minlength=self.Nk1)             # this slice's update to the numerator of the ensemble average
                current_par_bin=self.parbin_indices_column[i]

                sum_modsq_T_tilde[current_par_bin,:]+= current_binsums  # update the numerator   of the ensemble avg
                N_modsq_T_tilde[  current_par_bin,:]+= slice_bin_counts # update the denominator of the ensemble avg
            
            sum_modsq_T_tilde_truncated= sum_modsq_T_tilde[:-1,:-1]
            N_modsq_T_tilde_truncated=   N_modsq_T_tilde[  :-1,:-1]

        N_modsq_T_tilde_truncated[N_modsq_T_tilde_truncated==0]=maxint # avoid nans

        avg_modsq_T_tilde=sum_modsq_T_tilde_truncated/N_modsq_T_tilde_truncated
        P=np.array(avg_modsq_T_tilde/self.Veff)
        self.P=P

    def interpolate_P(self,k_want,avoid_extrapolation=True):
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
        if (self.k1bins is not None):
            kpar_have_lo=  self.k0bins[0]
            kpar_have_hi=  self.k0bins[-1]
            kperp_have_lo= self.k1bins[0]
            kperp_have_hi= self.k1bins[-1]

            self.k0bins_interp,self.k1bins_interp=k_want
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
            self.P_interp=interpn((self.k0bins,self.k1bins),self.P,(self.k0_interp_grid,self.k1_interp_grid),method="cubic",bounds_error=avoid_extrapolation,fill_value=None)
        else:
            self.k0bins_interp=k_want
            k_have_lo=self.k0bins[0]
            k_have_hi=self.k1bins[-1]
            k_want_lo=self.k0bins_interp[0]
            k_want_hi=self.k0bins_interp[-1]
            if (k_want_lo<k_have_lo):
                extrapolation_warning("low k",k_want_lo,k_have_lo)
            if (k_want_hi>k_have_hi):
                extrapolation_warning("high k",k_want_hi,k_have_hi)
            P_interpolator=interp1d(self.k0bins,self.P,kind="cubic",bounds_error=avoid_extrapolation,fill_value="extrapolate")
            self.P_interp=P_interpolator(self.k0bins_interp)
        
        def generate_box(self):
            assert(self.Nvox>=self.Nk0), "Nvox>=Nbins is baked into the code for now. I'll circumvent this with interpolation later, but, for now, do you really even want Nvox<Nbins?"
            if (self.P==None):
                raise NotEnoughInfoError
            if (self.Nk1>0):
                raise UnsupportedBinningMode # for now, I can only generate a box from a spherically binned power spectrum
            # not warning abt potentially overwriting T -> the only case where info would be lost is where self.P==None, and I already have a separate warning for that
            
            sigmas=np.sqrt(self.Veff*self.P/2) # from inverting the estimator equation and turning variances into std devs
            sigmas=np.reshape(sigmas,(self.Nbins,))
            T_tilde_Re=np.zeros((self.Nvox,self.Nvox,self.Nvox))
            T_tilde_Im=np.zeros((self.Nvox,self.Nvox,self.Nvox))

            for i,sig in enumerate(sigmas):
                i_eq_bin_indices=i==self.sph_bin_indices              # mask: voxels that fall into the bin under consideration during this iteration
                here=np.nonzero(i_eq_bin_indices)                     # all box indices where the corresponding bin index is the ith bin floor (iterable)
                num_here=len(np.argwhere(i_eq_bin_indices))           # number of voxels in the bin currently under consideration
                samps_Re=np.random.normal(scale=sig,size=(num_here,)) # samples for filling the current bin
                samps_Im=np.random.normal(scale=sig,size=(num_here,))
                if (num_here>0):
                    T_tilde_Re[here]=samps_Re
                    T_tilde_Im[here]=samps_Im
            
            T_tilde=T_tilde_Re+1j*T_tilde_Im # have not yet applied the symmetry that ensures T is real-valued 
            T=fftshift(irfftn(T_tilde*self.d3k,s=(self.Nvox,self.Nvox,self.Nvox),axes=(0,1,2),norm="forward"))/(twopi)**3 # handle in one line: shiftedness, ensuring T is real-valued and box-shaped, enforcing the cosmology Fourier convention
            self.T=T

        def avg_realizations(self):
            assert(self.P is not None), ""