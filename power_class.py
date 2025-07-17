import numpy as np
from scipy.interpolate import interpn,interp1d
from numpy.fft import fftshift,ifftshift,fftn,irfftn,fftfreq

"""
this module helps connect ensemble-averaged power spectrum estimates and 
cosmological brighness temperature boxes for assorted interconnected use cases:
1. generate a power spectrum that describes the statistics of a cosmo box
2. generate realizations of a cosmo box consistent with a known power spectrum
3. iterate power spec calcs from different box realizations until convergence
4. interpolate a power spectrum
"""

pi=np.pi
twopi=2.*pi
maxfloat= np.finfo(np.float64).max
maxint=   np.iinfo(np.int64  ).max
nearly_zero=1./maxfloat

class ResolutionError(Exception):
    pass
class UnsupportedBinningMode(Exception):
    pass
class UnsupportedPrimaryBeamType(Exception):
    pass
class NotEnoughInfoError(Exception):
    pass
class NotYetImplementedError(Exception):
    pass
class ConflictingInfoError(Exception): # make these inherit from each other (or something) to avoid repetitive code
    pass
def extrapolation_warning(regime,want,have):
    print("WARNING: if extrapolation is permitted in the interpolate_P call, it will be conducted for {:15s} (want {:9.4}, have{:9.4})".format(regime,want,have))
    return None

class cosmo_stats(object):
    def __init__(self,
                 Lsurvey,                                                                # nonnegotiable for box->spec and spec->box calcs
                 T=None,P_fid=None,Nvox=None,                                            # need one of either T or P to get started; I also check for any conflicts with Nvox
                 primary_beam=None,primary_beam_args=None,primary_beam_type="Gaussian",  # primary beam considerations
                 Nk0=10,Nk1=0,binning_mode="lin",                                        # binning considerations for power spec realizations
                 realization_ceiling=50,frac_tol=0.05,                                   # max number of realizations
                 k0bins_interp=None,k1bins_interp=None,                                  # bins where it would be nice to know about P_converged
                 P_realizations=None,P_converged=None                                    # power spectra related to averaging over those from dif box realizations
                 ):                                                                      # implement later: synthesized beam considerations
        """
        Lsurvey             :: float                      :: side length of cosmo box       :: Mpc
        T                   :: (Nvox,Nvox,Nvox) of floats :: cosmo box                      :: K
        P_fid               :: (Nk0_fid,) of floats       :: sph binned fiducial power spec :: K^2 Mpc^3
        Nvox                :: float                      :: # voxels PER SIDE of cosmo box :: ---
        primary_beam        :: callable                   :: power beam in Cartesian coords :: ---
        primary_beam_args   :: tuple of floats            :: Gaussian: "μ"s and "σ"s        :: Gaussian: sigLoS, r0 in Mpc; fwhm_x, fwhm_y in rad
        primary_beam_type   :: string                     :: implement later: Airy etc.     :: ---
        Nk0, Nk1            :: int                        :: # power spec bins for axis 0/1 :: ---
        realization_ceiling :: int                        :: max # of realiz in p.s. avg    :: ---
        frac_tol            :: float                      :: max fractional amount by which :: ---
                                                             the p.s. avg can change w/ the 
                                                             addition of the latest realiz. 
                                                             and the ensemble average is 
                                                             considered converged
        k0bins_interp,      :: (Nk0_interp,) of floats    :: bins to which to interpolate   :: 1/Mpc
        k1bins_interp          (Nk1_interp,) of floats       the converged power spec (prob
                                                             set by survey considerations)
        P_realizations      :: if Nk1==0: (Nk0,) floats   :: sph/cyl power specs for dif    :: K^2 Mpc^3
                               if Nk1>0: (Nk0,Nk1) floats    realizations of a cosmo box 
        P_converged         :: same as that of P_fid      :: average of P_realizations      :: idem ^
        
        
        """
        # spectrum and box
        self.Lsurvey=Lsurvey
        self.P_fid=P_fid
        if ((T is None) and (P_fid is None)):                           # must start with either a box or a fiducial power spec
            raise NotEnoughInfoError
        elif ((P_fid is not None) and (T is None) and (Nvox is None)): # to generate boxes from a power spec, il faut some way of determining how many voxels to use per side
            raise NotEnoughInfoError
        else:                                                           # there is possibly enough info to proceed, but still need to check for conflicts
            self.T=T
            if ((Nvox is not None) and (T is not None)):                # possible conflict: if both Nvox and a box are passed, 
                if (Nvox!=T.shape[0]):                                  # but Nvox and the box shape disagree,
                    raise ConflictingInfoError                          # estamos en problemas
                else:
                    self.Nvox=self.T.shape[0]                           # otherwise, initialize the Nvox attribute
            elif (Nvox is not None):                                    # if Nvox was passed but T was not, use Nvox to initialize the Nvox attribute
                self.Nvox=Nvox
            else:                                                       # remaining case: T was passed but Nvox was not, so use the shape of T to initialize the Nvox attribute
                self.Nvox=self.T.shape[0]
            
            if (P_fid is not None): # no hi fa res si the fiducial power spectrum has a different dimensionality or bin width than the realizations you plan to generate
                Pfidshape=P_fid.shape
                Pfiddims=len(Pfidshape)
                if (Pfiddims==2):
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

        # Fourier space
        self.Deltak=twopi/self.Lsurvey                                  # voxel side length
        self.d3k=self.Deltak**3                                         # volume element / voxel volume
        self.k_vec_for_box_corner=twopi*fftfreq(self.Nvox,d=self.Delta) # one Cartesian coordinate axis - non-fftshifted/ corner origin
        self.k_vec_for_box_centre=fftshift(self.k_vec_for_box_corner)   # one Cartesian coordinate axis -     fftshifted/ centre origin
        self.kx_grid_corner,self.ky_grid_corner,self.kz_grid_corner=np.meshgrid(self.k_vec_for_box_corner,
                                                                                self.k_vec_for_box_corner,
                                                                                self.k_vec_for_box_corner,
                                                                                indexing="ij")              # box-shaped Cartesian coords
        self.kx_grid_centre,self.ky_grid_centre,self.kz_grid_centre=np.meshgrid(self.k_vec_for_box_centre,
                                                                                self.k_vec_for_box_centre,
                                                                                self.k_vec_for_box_centre,
                                                                                indexing="ij")
        
        # sensible bins according to the limits of the box
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
        self.k_grid_corner=               np.sqrt(self.kx_grid_corner**2+self.ky_grid_corner**2+self.kz_grid_corner**2)   # k magnitudes for each voxel
        self.sph_bin_indices_corner=      np.digitize(self.k_grid_corner,self.k0bins)                                     # sph bin that each voxel falls into
        self.k_grid_centre=               np.sqrt(self.kx_grid_centre**2+self.ky_grid_centre**2+self.kz_grid_centre**2)   # same thing but fftshifted/ centre-origin
        self.sph_bin_indices_centre=      np.digitize(self.k_grid_centre,self.k0bins)
        self.sph_bin_indices_1d_corner=   np.reshape(self.sph_bin_indices_corner, (self.Nvox**3,)) # 1d version of ^ (compatible with np.bincount)
        self.sph_bin_indices_1d_centre=   np.reshape(self.sph_bin_indices_centre, (self.Nvox**3,))

            # voxel grids for cyl binning
        if (self.Nk1>0):
            self.kpar_column_corner= np.abs(self.k_vec_for_box_corner)                                          # magnitudes of kpar for a representative column along the line of sight (z-like)
            self.kperp_slice_corner= np.sqrt(self.kx_grid_corner**2+self.ky_grid_corner**2)[:,:,0]              # magnitudes of kperp for a representative slice transverse to the line of sight (x- and y-like)
            self.perpbin_indices_slice_corner=    np.digitize(self.kperp_slice_corner,self.k1bins)              # cyl kperp bin that each voxel falls into
            self.perpbin_indices_slice_1d_corner= np.reshape(self.perpbin_indices_slice_corner,(self.Nvox**2,)) # 1d version of ^ (compatible with np.bincount)
            self.parbin_indices_column_corner=    np.digitize(self.kpar_column_corner,self.k0bins)              # cyl kpar bin that each voxel falls into

            self.kpar_column_centre= np.abs(self.k_vec_for_box_centre)
            self.kperp_slice_centre= np.sqrt(self.kx_grid_centre**2+self.ky_grid_centre**2)[:,:,0]
            self.perpbin_indices_slice_centre=    np.digitize(self.kperp_slice_centre,self.k1bins)
            self.perpbin_indices_slice_1d_centre= np.reshape(self.perpbin_indices_slice_centre,(self.Nvox**2,))
            self.parbin_indices_column_centre=    np.digitize(self.kpar_column_centre,self.k0bins)

        # primary beam
        self.primary_beam=primary_beam
        self.primary_beam_args=primary_beam_args
        self.primary_beam_type=primary_beam_type
        if (self.primary_beam is not None): # non-identity primary beam
            if self.primary_beam_type=="Gaussian":
                self.sigLoS,self.fwhm_x,self.fwhm_y,self.r0=self.primary_beam_args
                self.evaled_response=self.primary_beam(self.xx_grid,self.yy_grid,self.zz_grid,self.sigLoS,self.fwhm_x,self.fwhm_y,self.r0) # custom_response(X,Y,Z,sigLoS,beamfwhm_x,beamfwhm_y,r0)
                self.Veff=np.sum(self.evaled_response*self.d3r)        # rectangular sum method
                self.evaled_response[self.evaled_response==0]=maxfloat # protect against division-by-zero errors
            else:
                UnsupportedPrimaryBeamType
        else:                               # identity primary beam
            self.Veff=self.Lsurvey**3
            self.evaled_response=np.ones((self.Nvox,self.Nvox,self.Nvox))
        
        # realization averaging
        self.realization_ceiling=realization_ceiling
        self.frac_tol=frac_tol

        # P_converged interpolation bins
        self.k0bins_interp=k0bins_interp
        self.k1bins_interp=k1bins_interp

        # realization, averaging, and interpolation placeholders if no prior info
        if (P_realizations is not None):       # maybe you want to import realizations from a prev run and just add more? (unclear why you'd have left the
            self.P_realizations=P_realizations # prev run w/o a converged average, unless, maybe, you want to re-run with a stricter convergence threshold?)
        else:
            self.P_realizations=[] 
        if (P_converged is not None):          # maybe you have a converged power spec average from a previous calc and just want to interpolate or generate more boxes?
            self.P_converged=P_converged
        else:
            self.P_converged=None
        self.P_interp=None                     # can't init with this because, if you had one, there'd be no point of using cosmo_stats (at best, you can provide a P_fid)

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
            
    def generate_P(self):
        """
        philosophy: 
        * compute the power spectrum of a known cosmological box and bin it spherically or cylindrically
        * append to the list of reconstructed P realizations (self.P_realizations)
        """
        if (self.T is None): # power spec has to come from a box
            self.generate_box()

        T_no_primary_beam=  self.T/self.evaled_response
        T_tilde=            fftshift(fftn((ifftshift(T_no_primary_beam)*self.d3r)))
        modsq_T_tilde=     (T_tilde*np.conjugate(T_tilde)).real

        if (self.Nk1==0): # bin to sph
            modsq_T_tilde_1d= np.reshape(modsq_T_tilde,    (self.Nvox**3,))

            sum_modsq_T_tilde= np.bincount(self.sph_bin_indices_1d_centre, 
                                           weights=modsq_T_tilde_1d, 
                                           minlength=self.Nk0)       # for the ensemble avg: sum    of modsq_T_tilde values in each bin
            N_modsq_T_tilde=   np.bincount(self.sph_bin_indices_1d_centre,
                                           minlength=self.Nk0)       # for the ensemble avg: number of modsq_T_tilde values in each bin
            sum_modsq_T_tilde_truncated=sum_modsq_T_tilde[:-1]       # excise sneaky corner modes: I devised my binning to only tell me about voxels w/ k<=(the largest sphere fully enclosed by the box), and my bin edges are floors. But, the highest floor corresponds to the point of intersection of the box and this largest sphere. To stick to my self-imposed "the stats are not good enough in the corners" philosophy, I must explicitly set aside the voxels that fall into the "catchall" uppermost bin. 
            N_modsq_T_tilde_truncated=  N_modsq_T_tilde[:-1]         # idem ^
        else:             # bin to cyl
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
                current_par_bin=self.parbin_indices_column[i]

                sum_modsq_T_tilde[current_par_bin,:]+= current_binsums  # update the numerator   of the ensemble avg
                N_modsq_T_tilde[  current_par_bin,:]+= slice_bin_counts # update the denominator of the ensemble avg
            
            sum_modsq_T_tilde_truncated= sum_modsq_T_tilde[:-1,:-1] # excise sneaky corner modes (see the analogous operation in the sph branch for an explanation)
            N_modsq_T_tilde_truncated=   N_modsq_T_tilde[  :-1,:-1] # idem ^

        N_modsq_T_tilde_truncated[N_modsq_T_tilde_truncated==0]=maxint # avoid division-by-zero errors during the division the estimator demands

        avg_modsq_T_tilde=sum_modsq_T_tilde_truncated/N_modsq_T_tilde_truncated
        P=np.array(avg_modsq_T_tilde/self.Veff)
        self.P_realizations.append(P) # prevent overwriting errors

    def interpolate_P(self,avoid_extrapolation=True):
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
            self.P_interp=interpn((self.k0bins,self.k1bins),self.P,(self.k0_interp_grid,self.k1_interp_grid),method="cubic",bounds_error=avoid_extrapolation,fill_value=None)
        else:
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
        """
        philosophy: generate a box that comprises a random realization of a known power spectrum
        """
        assert(self.Nvox>=self.Nk0), "Nvox>=Nbins is baked into the code for now. I'll circumvent this with interpolation later, but, for now, do you really even want Nvox<Nbins?"
        if (self.P_fid is None):
            raise NotEnoughInfoError
        if (self.Nk1>0):
            raise UnsupportedBinningMode # for now, I can only generate a box from a spherically binned power spectrum
        # not warning abt potentially overwriting T -> the only case where info would be lost is where self.P_fid is None, and I already have a separate warning for that
        
        sigmas=np.sqrt(self.Veff*self.P_fid/2.) # from inverting the estimator equation and turning variances into std devs
        print("generate_box: sigmas=",sigmas)
        sigmas=np.reshape(sigmas,(self.fid_Nk0,))
        T_tilde_Re=np.zeros((self.Nvox,self.Nvox,self.Nvox))
        T_tilde_Im=np.zeros((self.Nvox,self.Nvox,self.Nvox))

        for i,sig in enumerate(sigmas):
            here=np.nonzero(i==self.sph_bin_indices_corner)              # all box indices where the corresponding bin index is the ith bin floor (iterable)
            num_here=len(np.argwhere(i==self.sph_bin_indices_corner))    # number of voxels in the bin currently under consideration
            samps_Re=np.random.normal(scale=sig,size=(num_here,)) # samples for filling the current bin
            samps_Im=np.random.normal(scale=sig,size=(num_here,))
            if (num_here>0):
                T_tilde_Re[here]=samps_Re
                T_tilde_Im[here]=samps_Im
        
        T_tilde=T_tilde_Re+1j*T_tilde_Im # have not yet applied the symmetry that ensures T is real-valued 
        T=fftshift(irfftn(T_tilde*self.d3k,s=(self.Nvox,self.Nvox,self.Nvox),axes=(0,1,2),norm="forward"))/(twopi)**3 # handle in one line: shiftedness, ensuring T is real-valued and box-shaped, enforcing the cosmology Fourier convention
        self.T=T

    def avg_realizations(self):
        assert(self.P_fid is not None), "cannot average over numerically windowed realizations without a fiducial power spec"
        self.not_converged=True
        i=0
        while (self.not_converged and i<self.realization_ceiling):
            self.generate_box()      # generates a new T realization from self.P_fid and stores it in self.T
            self.generate_P()        # appends a P realization to self.P_realizations
            self.check_convergence() # updates self.not_converged
            i+=1

        arr_realiz_holder=np.array(self.P_realizations)
        if (i>1):
            self.P_converged=np.mean(arr_realiz_holder,axis=-1)
        else:
            self.P_converged=np.reshape(arr_realiz_holder,(self.fid_Nk0,))
        self.num_realiz_evaled=i # not called anything to the effect of "num_realiz_converged" bc it might not have converged if i hit the realization ceiling

    def check_convergence(self):
        """
        figure_of_merit is the ratio between the sample stddevs for ensembles containing the (0th through [n-1]st) and (0th through nth) realizations
        if the ensemble average has converged, adding the nth realization shouldn't change the variance that much--hence examining the ratio
        (but the scientifically rigorous thing to compare is the std dev of mean, which has the extra 1/sqrt(i))
        """
        arr_realiz_holder=np.array(self.P_realizations)
        arr_realiz_holder_shape=arr_realiz_holder.shape
        _,_,n=arr_realiz_holder_shape
        ndims=len(arr_realiz_holder_shape)
        prefac=np.sqrt((n-1)/n)
        if ndims==2:
            figure_of_merit=prefac*np.std(arr_realiz_holder[0,:],ddof=1)/np.std(arr_realiz_holder[0,:-1],ddof=1)     # cosmic variance dominates the scatter between realizations, so focus on that, instead of more specific (and computationally intensive...) .any() or .all() comparisons
        elif ndims==3:
            figure_of_merit=prefac*np.std(arr_realiz_holder[0,0,:],ddof=1)/np.std(arr_realiz_holder[0,0,:-1],ddof=1) # idem
        else:
            raise NotYetImplementedError
        self.not_converged=(figure_of_merit<self.frac_tol)