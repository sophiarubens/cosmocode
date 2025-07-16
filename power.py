import numpy as np
from scipy.interpolate import interpn,interp1d
# from scipy.integrate import tplquad
# import time
# from bias_helper_fcns import custom_response2
from numpy.fft import fftshift,ifftshift,fftn,irfftn,fftfreq

pi=np.pi
twopi=2.*pi
maxint=  np.iinfo(np.int64  ).max
maxfloat=np.finfo(np.float64).max
eps=1e-30

"""
this module helps connect ensemble-averaged power spectrum estimates 
and cosmological brighness temperature boxes for three main use cases:
1. generate a power spectrum that describes the statistics of a cosmo box
2. generate realizations of a cosmo box consistent with a known power spectrum
3. interpolate a power spectrum
"""

class ResolutionError(Exception):
    pass

def get_Veff(custom_estimator,custom_estimator_args,Lsurvey,Nvox):

    # using rectangles/sums
    vec=Lsurvey*fftshift(fftfreq(Nvox))
    xgrid,ygrid,zgrid=np.meshgrid(vec,vec,vec,indexing="ij")
    sigLoS,beamfwhm_x,beamfwhm_y,r0=custom_estimator_args
    evaled_response=custom_estimator(xgrid,ygrid,zgrid,sigLoS,beamfwhm_x,beamfwhm_y,r0)
    d3r=(Lsurvey/Nvox)**3
    Veff=np.sum(evaled_response*d3r)
    
    # # using scipy
    # bound=Lsurvey/2.
    # Veff,_= tplquad(custom_response2,-bound,bound,-bound,bound,-bound,bound,args=custom_estimator_args) # hackily global from other file... obv do not leave this way if I revert to this calc strategy
    return Veff,evaled_response

def P_driver(T, k, Lsurvey, primary_beam=False,primary_beam_args=False):
    """
    philosophy:
    * generate a power spectrum consistent with a brightness temperature box for which you already know the k-bins
    * In practice, to obtain a power spectrum you should access this routine through a wrapper that calculates (for now, lin- or log-spaced bins) corresponding to a particular Lsurvey, Nk0, (and, if relevant, Nk1), and Nvox case. 
      If instrument considerations (channel width and baseline distribution) lead you to be interested in the power spectrum at some other k-modes, you should interpolate the output of generate_P

    inputs:
    T                = Nvox x Nvox x Nvox data box for which you wish to create the power spectrum
    k                = k-bins for power spectrum indexing ... designed to be provided by one of two wrapper functions: user-created (custom) bins or automatically generated (linear- or log-spaced) bins
                         - shape (Nk,) or similar (e.g. (Nk,1) or (1,Nk)) for spherical binning
                         - shape [(Nkpar,),(Nkperp,)] or similar for cylindrical binning
    Lsurvey          = side length of the cosmological box (Mpc) (just sets the volume element scaling... nothing to do with pixelization)
    custom_estimator = if not False, a callable defined in CARTESIAN coordinates to integrate from -Lsurvey/2 to Lsurvey/2 along all three axes to get the estimator denominator for the case of a nontrivial instrument response
    custom_estimator_args = if not None, the "parameters" (i.e. not variables) in custom_estimator; realistically, (sigLoS,beamfwhm1,beamfwhm2,r0,)

    outputs:
    k-modes and powers of the power spectrum
         - if binto=="sph": [(Nk,),(Nk,)] object unpackable as kbins,powers
         - if binto=="cyl": [[(Nkpar,),(Nkperp,)],(Nkpar,Nkperp)] object unpackable as [kparbins,kperpbins],powers
    """
    # setup
    lenk=len(k)
    if lenk==2:
        binto="cyl"
    else:
        binto="sph"
    Nvox  = T.shape[0]
    Delta = Lsurvey/Nvox  # voxel side length
    d3r   = Delta**3      # voxel volume
    if not primary_beam:  # identity primary beam
        Veff=Lsurvey**3
        evaled_response=np.ones((Nvox,Nvox,Nvox))
    else:                 # non-identity primary beam
        Veff,evaled_response=get_Veff(primary_beam,primary_beam_args,Lsurvey,Nvox)
        evaled_response[evaled_response==0]=maxfloat
        # overwrite_condition=evaled_response==0
        # num_to_overwrite=np.sum(overwrite_condition)
        # draws_to_overwrite_with=np.random.randn(num_to_overwrite)
        # evaled_response[overwrite_condition]=draws_to_overwrite_with

    # print("P_driver: Veff=", Veff)

    # process the box values
    T_no_primary_beam=T/evaled_response
    T_tilde=        fftshift(fftn((ifftshift(T_no_primary_beam)*d3r)))
    modsq_T_tilde= (T_tilde*np.conjugate(T_tilde)).real

    # establish Fourier duals to box coordinates (Cartesian paradigm)
    k_vec_for_box= twopi*fftshift(fftfreq(Nvox,d=Delta)) 
    kx_box_grid,ky_box_grid,kz_box_grid= np.meshgrid(k_vec_for_box,k_vec_for_box,k_vec_for_box,indexing="ij") # centre-origin Fourier-space grid

    if (binto=="sph"):
        Nk=lenk # number of k-modes to put in the power spectrum

        # prepare to tie the processed box values to relevant k-values
        k_box=          np.sqrt(kx_box_grid**2+ky_box_grid**2+kz_box_grid**2) # scalar k for each voxel
        bin_indices=    np.digitize(k_box,k)                                  # box with entries indexing which bin each voxel belongs in [DEFAULT BEHAVIOUR IS RIGHT==FALSE] 
        bin_indices_1d=   np.reshape(bin_indices,(Nvox**3,))       # to bin, I use np.bincount, which requires 1D input
        modsq_T_tilde_1d= np.reshape(modsq_T_tilde,    (Nvox**3,)) # ^ same preprocessing

        # binning
        sum_modsq_T_tilde= np.bincount(bin_indices_1d,weights=modsq_T_tilde_1d,minlength=Nk) # for the ensemble average: sum    of modsq_T_tilde values in each bin
        N_modsq_T_tilde=   np.bincount(bin_indices_1d,                         minlength=Nk) # for the ensemble average: number of modsq_T_tilde values in each bin
        sum_modsq_T_tilde_truncated=sum_modsq_T_tilde[:-1]
        N_modsq_T_tilde_truncated=N_modsq_T_tilde[:-1]

        # ##
        # # try to manually override lost power... might not be as mathematically motivated as I'd hope
        # N_voxels_per_bin=np.zeros(Nk)
        # N_voxels_per_bin_masked=np.zeros(Nk)
        # bin_indices_masked=np.copy(bin_indices)            # start with the original bin indices
        # bin_indices_masked[evaled_response==maxfloat]=Nk+1 # rename voxels where the primary beam was originally effectively zero s.t. they will not be counted in the Nk loop
        # for i in range(Nk):
        #     N_voxels_per_bin[i]=       len(np.unique(k_box[bin_indices==       i]))
        #     N_voxels_per_bin_masked[i]=len(np.unique(k_box[bin_indices_masked==i]))
        # # print("N_voxels_per_bin=       ",N_voxels_per_bin)
        # # print("N_voxels_per_bin_masked=",N_voxels_per_bin_masked)
        # # print("N_voxels_per_bin_masked/N_voxels_per_bin=",N_voxels_per_bin_masked/N_voxels_per_bin)
        # underest_factor=N_voxels_per_bin_masked/N_voxels_per_bin
        # sum_modsq_T_tilde_truncated*=(underest_factor**2)
        # ##

    elif (binto=="cyl"): # kpar is z-like
        kpar,kperp=k # kpar being unpacked first here DOES NOT change my treatment of kpar as a z-like coordinate in 3D arrays (look at these lines to re-convince myself: kperpmags=, modsq_T_tilde_slice=,...)
        Nkpar=len(kpar)
        Nkperp=len(kperp)

        # prepare to tie the processed box values to relevant k-values
        kperpmags=                np.sqrt(kx_box_grid**2+ky_box_grid**2)       # here, I'm jumping on the "kpar is like z" bandwagon,, probably fix and avoid mixing conventions at some point
        kperpmags_slice=          kperpmags[:,:,0]                             # take a representative slice, now that I've rigorously checked that things vary the way I want
        perpbin_indices_slice=    np.digitize(kperpmags_slice,kperp)           # each representative slice has the same bull's-eye pattern of bin indices... no need to calculate for each slice, not to mention how it would be overkill to reshape the whole box down to 1D and digitize and bincount that
        perpbin_indices_slice_1d= np.reshape(perpbin_indices_slice,(Nvox**2,)) # even though I've chosen a representative slice, I still need to flatten down to 1D in anticipation of bincounting
        kparmags_column=          np.abs(k_vec_for_box)                        # to avoid the negative kpar/kz half of the box from being all shuffled into bin 0 (pos/neg issue)
        parbin_indices_column=    np.digitize(kparmags_column,kpar)            # vector with entries indexing which kpar bin each voxel belongs in (pending slight postprocessing in the loop) ... just as I could look at a representative slice for the kperp direction, I can look at a representative chunk for the LoS direction (though, naturally, in this case it is a "column") ... no need to reshape, b/c (1). it's already 1D and (2). I don't have an explicit bincount call along this axis because I iterate over kpar slices
        
        # binning 
        sum_modsq_T_tilde= np.zeros((Nkpar+1,Nkperp+1)) # for the ensemble average: sum    of modsq_T_tilde values in each bin  ... each time I access it, I'll access the kparBIN row of interest, but update all NkperpBIN columns
        N_modsq_T_tilde=   np.zeros((Nkpar+1,Nkperp+1)) # for the ensemble average: number of modsq_T_tilde values in each bin

        for i in range(Nvox): # iterate over the kpar axis of the box to capture all LoS slices
            if (i==0): # stats of the kperp "bull's eye" slice
                slice_bin_counts=  np.bincount(perpbin_indices_slice_1d, minlength=Nkperp) # each slice's update to the denominator of the ensemble average
            modsq_T_tilde_slice=    modsq_T_tilde[:,:,i]                                                                  # take the slice of interest of the preprocessed box values !! still treating kpar as z-like
            modsq_T_tilde_slice_1d= np.reshape(modsq_T_tilde_slice,(Nvox**2,))                                            # reshape to 1D for bincount compatibility
            current_binsums=        np.bincount(perpbin_indices_slice_1d,weights=modsq_T_tilde_slice_1d,minlength=Nkperp) # this slice's update to the numerator of the ensemble average
            current_par_bin=        parbin_indices_column[i]

            sum_modsq_T_tilde[current_par_bin,:]+= current_binsums  # update the numerator   of the ensemble average
            N_modsq_T_tilde[current_par_bin,:]+=   slice_bin_counts # update the denominator of the ensemble average
        
        sum_modsq_T_tilde_truncated= sum_modsq_T_tilde[:-1,:-1]
        N_modsq_T_tilde_truncated=   N_modsq_T_tilde[  :-1,:-1]

    else:
        assert(1==0), "only spherical and cylindrical power spectrum binning are currently supported"
        return None
    
    # avoid nans
    empty=np.nonzero(N_modsq_T_tilde_truncated==0)
    N_modsq_T_tilde_truncated[empty]=maxint

    # translate to power spectrum terms
    avg_modsq_T_tilde=sum_modsq_T_tilde_truncated/N_modsq_T_tilde_truncated
    P=np.array(avg_modsq_T_tilde/Veff)
    return [k,P]

def get_bins(Nvox,Lsurvey,Nk,mode):
    Delta=Lsurvey/Nvox
    kmax=pi/Delta # to handle the +/-
    kmin=twopi/Lsurvey

    if (mode=="log"):
        kbins=np.logspace(np.log10(kmin),np.log10(kmax),num=Nk)
        limiting_spacing=twopi*(10.**(kmax)-10.**(kmax-(np.log10(Nvox)/Nk)))
    elif (mode=="lin"):
        kbins=np.linspace(kmin,kmax,Nk)
        limiting_spacing=twopi*(0.5*Nvox-1)/(Nk) # version for a kmax that is "aware that" there are +/- k-coordinates in the box
    else:
        assert(1==0), "only log and linear binning are currently supported"
    return kbins,limiting_spacing

def generate_P(T, mode, Lsurvey, Nk0, Nk1=0, primary_beam=False,primary_beam_args=False):
    """
    philosophy:
    * generate a spherically or cylindrically binned power spectrum with lin- or log-spaced bins, consistent with a given brightness temperature box
    * wrapper function for the power spectrum function P_driver(T, k, Lsurvey, Nvox) 

    inputs:
    T                = Nvox x Nvox x Nvox data box for which you wish to create the power spectrum
    mode             = binning mode (linear or logarithmic)
    Lsurvey          = side length of the cosmological box (Mpc)
    Nk0              = number of k-bins to include in the power spectrum (if this is the only nonzero Nk, the power spectrum will be binned spherically)
    Nk1              = number of k-bins to include along axis=1 of the power spectrum (if nonzero, the power spectrum will be binned cylindrically)
    custom_estimator = if not False, a callable defined in CARTESIAN coordinates to integrate from -Lsurvey/2 to Lsurvey/2 along all three axes to get the estimator denominator for the case of a nontrivial instrument response
    
    outputs:
    one copy of P_driver() output
    """
    Nvox=T.shape[0]
    deltak_box=twopi/Lsurvey

    k0bins,limiting_spacing_0=get_bins(Nvox,Lsurvey,Nk0,mode)
    if (limiting_spacing_0<deltak_box):
        raise ResolutionError
    
    if (Nk1>0):
        k1bins,limiting_spacing_1=get_bins(Nvox,Lsurvey,Nk1,mode)
        if (limiting_spacing_1<deltak_box):
            raise ResolutionError
        kbins=[k0bins,k1bins]
    else:
        kbins=k0bins
    return P_driver(T,kbins,Lsurvey,primary_beam=primary_beam,primary_beam_args=primary_beam_args)

def P_avg_over_realizations(T,mode,Lsurvey,Nk0,Nk1=0,primary_beam=False,primary_beam_args=False,Nrealiz=50,fractol=0.05):
    """
    philosophy:
    * compute realizations of a power spectrum and compute the ensemble average
    * keep adding realizations until either 
        (a) the scatter between realizations has converged to within the specified fractional tolerance
        (b) the number of computed realizations reaches the specified ceiling
    """
    realization_holder=[]
    not_converged=True
    i=0
    while (not_converged and i<Nrealiz):
        k,P=generate_P(T, mode, Lsurvey, Nk0, Nk1=Nk1,primary_beam=primary_beam,primary_beam_args=primary_beam_args)
        realization_holder.append(P)
        not_converged=check_convergence(realization_holder,fractol=fractol)
        i+=1
    
    arr_realiz_holder=np.array(realization_holder)
    print("arr_realiz_holder.shape=",arr_realiz_holder.shape)
    if i>1:
        avg_over_realizations=np.mean(arr_realiz_holder,axis=-1)
    else:
        avg_over_realizations=np.reshape(arr_realiz_holder,P.shape)
    return [k,avg_over_realizations],i

def check_convergence(realization_holder,fractol=0.05): # clear candidate for minimizing redundancy by making all the initializations class attributes instead of things shuffled around between functions, but... first, I need to get all my code working :,)
    arr_realiz_holder=np.array(realization_holder)
    realiz_holder_shape=arr_realiz_holder.shape
    n=realiz_holder_shape[-1] # sph binning: shape will be (Nk,Nrealiz_so_far); cyl binning: shape will be (Nkpar,Nkperp,Nrealiz_so_far)
    ndims=len(realiz_holder_shape)
    prefac=np.sqrt((n-1)/n)
    if ndims==2: # both branches: figure_of_merit is the ratio between the sample stddevs for ensembles containing the (0th through [n-1]st) and (0th through nth) realizations... if the ensemble average has converged, adding the nth realization shouldn't change the variance that much--hence examining the ratio
        figure_of_merit=prefac*np.std(arr_realiz_holder[0,:],ddof=1)/np.std(arr_realiz_holder[0,:-1],ddof=1)     # cosmic variance dominates the scatter between realizations, so focus on that, instead of more specific (and computationally intensive...) .any() or .all() comparisons
    elif ndims==3:
        figure_of_merit=prefac*np.std(arr_realiz_holder[0,0,:],ddof=1)/np.std(arr_realiz_holder[0,0,:-1],ddof=1) # idem
    else:
        assert(1==0), "binning strat not recognized (too many/few dims) --"+str(ndims)
    return figure_of_merit<fractol

def interpolate_P(P_have,k_have,k_want,avoid_extrapolation=True):
    """
    philosophy:
    * wraps scipy.interpolate.interpn
    * default behaviour upon requesting extrapolation: "ValueError: One of the requested xi is out of bounds in dimension 0"
    * if extrapolation is acceptable for your purposes, rerun with extrapolate=True (bounds_error supersedes fill_value, so there's no issue with fill_value always being set to what it needs to be to permit extrapolation [None for the nd case, "extrapolate" for the 1d case])

    inputs:
    P_have      = power spectrum you'd like to interpolate
                    * if sph: (Nk,) or equiv.
                    * if cyl: (Nkpar,Nkperp) or equiv.
    k_have      = k-modes at which you have power spectrum values
                    * if sph: (Nk,) or equiv.
                    * if cyl: ((Nkpar,),(Nkperp,)) or equiv.
    k_want      = k-modes at which you'd like to comment on the power spectrum (NOT necessary that Nk==Nk_want (sph) or Nkpar==Nkpar_want && Nkperp==Nkperp_want (cyl))
                    * if sph: (Nk_want,) or equiv.
                    * if cyl: ((Nkpar_want,),(Nkperp_want,)) or equiv. 
    extrapolate = boolean (would you like the wrapper to extrapolate if k_want is a superset of k_have?)

    outputs: 
    same format as the output of generate_P (which itself returns one copy of P_driver output)
    """
    if (len(k_have)==2): # still relying on the same somewhat hacky litmus test for sph vs. cyl as in generate_P (hacky because it is contingent on shuffling around the k-modes the way I have been)
        kpar_have,kperp_have=k_have
        kpar_have_lo=kpar_have[0]
        kpar_have_hi=kpar_have[-1]
        kperp_have_lo=kperp_have[0]
        kperp_have_hi=kperp_have[-1]

        kpar_want,kperp_want=k_want
        kpar_want_lo=kpar_want[0]
        kpar_want_hi=kpar_want[-1]
        kperp_want_lo=kperp_want[0]
        kperp_want_hi=kperp_want[-1]

        if (kpar_want_lo<kpar_have_lo):
            extrapolation_warning("low kpar",   kpar_want_lo,  kpar_have_lo)
        if (kpar_want_hi>kpar_have_hi):
            extrapolation_warning("high kpar",  kpar_want_hi,  kpar_have_hi)
        if (kperp_want_lo<kperp_have_lo):
            extrapolation_warning("low kperp",  kperp_want_lo, kperp_have_lo)
        if (kperp_want_hi>kperp_have_hi):
            extrapolation_warning("high kperp", kperp_want_hi, kperp_have_hi)
        kpar_want_grid,kperp_want_grid=np.meshgrid(kpar_want,kperp_want,indexing="ij")
        P_want=interpn((kpar_have,kperp_have),P_have,(kpar_want_grid,kperp_want_grid),method="cubic",bounds_error=avoid_extrapolation,fill_value=None)
    else:
        k_want_lo=k_want[0]
        k_want_hi=k_want[-1]
        k_have_lo=k_have[0]
        k_have_hi=k_have[-1]
        if (k_want_lo<k_have_lo):
            extrapolation_warning("low k",k_want_lo,k_have_lo)
        if (k_want_hi>k_have_hi):
            extrapolation_warning("high k",k_want_hi,k_have_hi)
        P_interpolator=interp1d(k_have,P_have,kind="cubic",bounds_error=avoid_extrapolation,fill_value="extrapolate")
        P_want=P_interpolator(k_want)
    return (k_want,P_want)

def extrapolation_warning(regime,want,have):
    print("WARNING: if extrapolation is permitted in interpolate_P call, it will be conducted for {:15s} (want {:9.4}, have{:9.4})".format(regime,want,have))
    return None 

def generate_box(P,k,Lsurvey,Nvox,primary_beam=False,primary_beam_args=False):
    """
    philosophy:
    generate a brightness temperature box consistent with a given matter power spectrum

    inputs:
    P = power spectrum
    k = Fourier space points at which the power spectrum is sampled
    Lsurvey = length of a cosmological box side, in Mpc 
    Nvox = number of voxels to create per cosmo box side

    outputs:
    Nvox x Nvox x Nvox brightness temp box
    """
    # helper variable setup
    k=k.real # enforce what makes sense physically
    P=P.real
    Nbins=len(P)
    assert(Nvox>=Nbins), "Nvox>=Nbins is baked into the code at the moment. I'm going to fix this (interpolation...) after I handle the more pressing issues, but for now, why would you even want Nvox<Nbins?"
    Delta=Lsurvey/Nvox
    d3k = (twopi/Lsurvey)**3 # Fourier-space voxel volume
    if not primary_beam:  # identity primary beam
        Veff=Lsurvey**3
    else:                 # non-identity primary beam
        Veff,_=get_Veff(primary_beam,primary_beam_args,Lsurvey,Nvox)

    # CORNER-origin k-grid
    k_vec_for_box=twopi*fftfreq(Nvox,d=Delta)
    KX,KY,KZ=np.meshgrid(k_vec_for_box,k_vec_for_box,k_vec_for_box)
    kgrid=np.sqrt(KX**2+KY**2+KZ**2)
    
    # take appropriate draws from normal distributions to populate T-tilde
    sigmas=np.sqrt(Veff*P/2)
    sigmas=np.reshape(sigmas,(Nbins,)) # transition from the (1,Nbins) of the CAMB PS to (Nbins,)
    T_tildere=np.zeros((Nvox,Nvox,Nvox))
    T_tildeim=np.zeros((Nvox,Nvox,Nvox))
    bin_indices=np.digitize(kgrid,k)

    for i,sig in enumerate(sigmas):
        here=np.nonzero(i==bin_indices)                      # all box indices where the corresponding bin index is the ith bin floor (iterable)
        numhere=len(np.argwhere(i==bin_indices))             # number of voxels in the bin currently under consideration
        sampsRe=np.random.normal(scale=sig, size=(numhere,)) # samples for filling the current bin
        sampsIm=np.random.normal(scale=sig, size=(numhere,))
        if (numhere>0):
            T_tildere[here]=sampsRe
            T_tildeim[here]=sampsIm

    T_tilde=T_tildere+1j*T_tildeim # no symmetries yet
    T=fftshift(irfftn(T_tilde*d3k,s=(Nvox,Nvox,Nvox),axes=(0,1,2),norm="forward"))/(2.*pi)**3 # applies the symmetries automatically! -> then return in user-friendly CENTRE-origin format (net zero shift when applied sequentially with generate_P)
    # T-=np.mean(T) # test: subtract off the monopole moment
    return kgrid,T,k_vec_for_box