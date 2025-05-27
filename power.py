import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interpn,interp1d
import time

pi=np.pi
twopi=2.*pi

"""
this module helps connect ensemble-averaged power spectrum estimates and cosmological brighness temperature boxes for three main use cases:
1. forward direction:   generate a power spectrum that describes the statistics of a brightness temperature box
2. backwards direction: generate one realization of a brightness temperature box consistent with a known power spectrum
3. regridding:          interpolate a power spectrum to k-modes of interest

call structure:
1. generate_P    -> 1. {{get_bins}}
                    2. {{P_driver}}
2. generate_box  ->    {{flip}}
3. interpolate_P

{{}} = generally no need to call directly; called as necessary by higher-level functions
"""

class ResolutionError(Exception):
    pass

def P_driver(T, k, Lsurvey):
    """
    philosophy:
    * generate a power spectrum consistent with a brightness temperature box for which you already know the k-bins
    * In practice, to obtain a power spectrum you should access this routine through a wrapper that calculates (for now, lin- or log-spaced bins) corresponding to a particular Lsurvey, Nk0, (and, if relevant, Nk1), and Nvox case. 
      If instrument considerations (channel width and baseline distribution) lead you to be interested in the power spectrum at some other k-modes, you should interpolate the output of generate_P

    inputs:
    T       = Nvox x Nvox x Nvox data box for which you wish to create the power spectrum
    k       = k-bins for power spectrum indexing ... designed to be provided by one of two wrapper functions: user-created (custom) bins or automatically generated (linear- or log-spaced) bins
                 - shape (Nk,) or similar (e.g. (Nk,1) or (1,Nk)) for spherical binning
                 - shape [(Nkpar,),(Nkperp,)] or similar for cylindrical binning
    Lsurvey = side length of the cosmological box (Mpc) (just sets the volume element scaling... nothing to do with pixelization)

    outputs:
    k-modes and powers of the power spectrum
         - if binto=="sph": [(Nk,),(Nk,)] object unpackable as kbins,powers
         - if binto=="cyl": [[(Nkpar,),(Nkperp,)],(Nkpar,Nkperp)] object unpackable as [kparvec,kperpvec],powers_on_grid **MAKE SURE THIS ENDS UP BEING TRUE**
    """
    # helper variables
    lenk=len(k)
    if lenk==2:
        binto="cyl"
    else: # kind of hackily relying on the properties of list-stored ragged arrays to let the "if" catch cases that need to be cyl and then shuffle everything else along to spherical ... not very robust against pathological calls
        binto="sph"
    Nvox  =T.shape[0]
    Delta = Lsurvey/Nvox # voxel side length
    dr3   = Delta**3     # voxel volume
    V     = Lsurvey**3   # volume of the cosmo box
    
    # process the box values
    Ts  = np.fft.ifftshift(T)*dr3 # T-ishifted (np wants a corner origin; ifftshift takes you there)
    Tts = np.fft.fftn(Ts)         # T-tilde
    Tt  = np.fft.fftshift(Tts)    # shift back to physics land
    mTt = Tt*np.conjugate(Tt)     # mod-squared of Tt
    mTt = mTt.real                # I checked, and there aren't even any machine precision issues in the imag part

    # establish Cartesian Fourier duals to box coordinates
    k_vec_for_box= twopi*np.fft.fftshift(np.fft.fftfreq(Nvox,d=Delta))
    kx_box_grid,ky_box_grid,kz_box_grid=      np.meshgrid(k_vec_for_box,k_vec_for_box,k_vec_for_box,indexing="ij") # centre-origin Fourier duals to config space coords (ofc !not !yet !binned)

    if (binto=="sph"):
        Nk=lenk # number of k-modes to put in the power spectrum

        # prepare to tie the processed box values to relevant k-values
        k_box=          np.sqrt(kx_box_grid**2+ky_box_grid**2+kz_box_grid**2) # scalar k for each voxel
        bin_indices=    np.digitize(k_box,k,right=False)                      # box with entries indexing which bin each voxel belongs in
        bin_indices_1d= np.reshape(bin_indices,(Nvox**3,))                    # to bin, I use np.bincount, which requires 1D input
        mTt_1d=         np.reshape(mTt,    (Nvox**3,))                        # ^ same preprocessing

        # binning
        summTt=np.bincount(bin_indices_1d,weights=mTt_1d,minlength=Nk) # for the ensemble average: sum    of mTt values in each bin
        NmTt=  np.bincount(bin_indices_1d,               minlength=Nk) # for the ensemble average: number of mTt values in each bin
        if (len(summTt)==(Nk+1)): # prune central voxel "below the floor of the lowest bin" extension bin
            summTt=summTt[1:]
            NmTt=  NmTt[1:]
        amTt=np.zeros(Nk) # template to store the ensemble average: to avoid division-by-zero errors, I use an empty-bin mask for the ensemble average sum/count division to leave zero power (instead of ending up with nan power) in empty bins

    elif (binto=="cyl"):
        kpar,kperp=k # this assumes my apparently-nontraditional convention of putting kpar first... fix this later, probably
        Nkpar=len(kpar)
        Nkperp=len(kperp)

        # prepare to tie the processed box values to relevant k-values
        kperpmags=                np.sqrt(kx_box_grid**2+ky_box_grid**2)         # here, I'm jumping on the "kpar is like z" bandwagon,, probably fix and avoid mixing conventions at some point
        kperpmags_slice=          kperpmags[:,:,0]                               # take a representative slice, now that I've rigorously checked that things vary the way I want
        perpbin_indices_slice=    np.digitize(kperpmags_slice,kperp,right=False) # each representative slice has the same bull's-eye pattern of bin indices... no need to calculate for each slice, not to mention how it would be overkill to reshape the whole box down to 1D and digitize and bincount that
        perpbin_indices_slice_1d= np.reshape(perpbin_indices_slice,(Nvox**2,))   # even though I've chosen a representative slice, I still need to flatten down to 1D in anticipation of bincounting
        parbin_indices_column=    np.digitize(k_vec_for_box,kpar, right=False)   # vector with entries indexing which kpar bin each voxel belongs in (pending slight postprocessing in the loop) ... just as I could look at a representative slice for the kperp direction, I can look at a representative chunk for the LoS direction (though, naturally, in this case it is a "column") ... no need to reshape, b/c (1). it's already 1D and (2). I don't have an explicit bincount call along this axis because I iterate over kpar slices

        # binning 
        summTt= np.zeros((Nkpar,Nkperp)) # for the ensemble average: sum    of mTt values in each bin  ... each time I access it, I'll access the kparBIN row of interest, but update all NkperpBIN columns
        NmTt=   np.zeros((Nkpar,Nkperp)) # for the ensemble average: number of mTt values in each bin
        for i in range(Nvox): # iterate over the kpar axis of the box to capture all LoS slices
            if (i==0): # stats of the kperp "bull's eye" slice
                slice_bin_counts= np.bincount(perpbin_indices_slice_1d, minlength=Nkperp) # each slice's update to the denominator of the ensemble average
                if (len(slice_bin_counts)==(Nkperp+1)): # prune central voxel "below the floor of the lowest bin" extension bin
                    slice_bin_counts=slice_bin_counts[1:]
            mTt_slice=       mTt[:,:,i]                                                                  # take the slice of interest of the preprocessed box values
            mTt_slice_1d=    np.reshape(mTt_slice,(Nvox**2,))                                            # reshape to 1D for bincount compatibility
            current_binsums= np.bincount(perpbin_indices_slice_1d,weights=mTt_slice_1d,minlength=Nkperp) # this slice's update to the numerator of the ensemble average
            if (len(current_binsums)==(Nkperp+1)): # prune central voxel "below the floor of the lowest bin" extension bin
                current_binsums=current_binsums[1:]
            current_par_bin= parbin_indices_column[i]
            if (current_par_bin!=0):                         # digitize philosophy is to use index 0 for values below the lowest bin floor (the origin voxel has kx,ky,kz=0,0,0, which yields a kpar and kperp smaller than the lowest bin floor, because a survey can never probe k=0 unless Lsurvey->infinity), so discard this point and move on
                current_par_bin-=           1                # other indices need to be shifted down by one because of how I treat bin floors
                summTt[current_par_bin,:]+= current_binsums  # update the numerator of the ensemble average
                NmTt[current_par_bin,:]+=   slice_bin_counts # update the denominator of the ensemble average
        amTt=np.zeros((Nkpar,Nkperp)) # template to store the ensemble average (same philosophy as for the spherical case—see above)
    else:
        assert(1==0), "only spherical and cylindrical power spectrum binning are currently supported"
        return None
    
    # translate to power spectrum terms
    nonemptybins=np.nonzero(NmTt)
    amTt[nonemptybins]=summTt[nonemptybins]/NmTt[nonemptybins]
    P=np.array(amTt/V)
    return [k,P]

def get_bins(Nvox,Lsurvey,Nk,mode):
    Delta=Lsurvey/Nvox
    kmax=twopi/Delta
    kmin=twopi/Lsurvey
    if (mode=="log"):
        kbins=np.logspace(np.log10(kmin),np.log10(kmax),num=Nk)
        arg=np.log(Nvox)/Nk
        limiting_spacing=twopi*(10.**(2.*arg)-10.**arg)
    elif (mode=="lin"):
        kbins=np.linspace(kmin,kmax,Nk)
        limiting_spacing=twopi*(Nvox-1)/(Nk*Lsurvey)
    else:
        assert(1==0), "only log and linear binning are currently supported"
    return kbins,limiting_spacing

def generate_P(T, mode, Lsurvey, Nk0, Nk1=0):
    """
    philosophy:
    * generate a spherically binned power spectrum with lin- or log-spaced bins, consistent with a given brightness temperature box
    * wrapper function for the power spectrum function P_driver(T, k, Lsurvey, Nvox) 

    inputs:
    T       = Nvox x Nvox x Nvox data box for which you wish to create the power spectrum
    mode    = binning mode (linear or logarithmic)
    Lsurvey = side length of the cosmological box (Mpc) (just sets the volume element scaling... nothing to do with pixelization)
    Nk0      = number of k-bins to include in the power spectrum (if this is the only nonzero Nk, the power spectrum will be binned spherically)
    Nk1     = number of k-bins to include along axis=1 of the power spectrum (if nonzero, the power spectrum will be binned cylindrically)

    outputs:
    one copy of P_driver() output
    """
    Nvox=T.shape[0]
    deltak_box=twopi/Lsurvey
    print("generate_P: Lsurvey,Nvox,Nk0,Nk1=",Lsurvey,Nvox,Nk0,Nk1)

    k0bins,limiting_spacing_0=get_bins(Nvox,Lsurvey,Nk0,mode) # get_bins(Nvox,Lsurvey,Nk,mode)
    if (limiting_spacing_0<deltak_box):
        raise ResolutionError
    
    if (Nk1>0):
        k1bins,limiting_spacing_1=get_bins(Nvox,Lsurvey,Nk1,mode)
        if (limiting_spacing_1<deltak_box):
            raise ResolutionError
        kbins=[k0bins,k1bins]
    else:
        kbins=k0bins
    return P_driver(T,kbins,Lsurvey)

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
    print("entering interpolate_P")
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
        P_want=interpn((kpar_have,kperp_have),P_have,(kpar_want_grid,kperp_want_grid),bounds_error=avoid_extrapolation,fill_value=None)
    else:
        P_interpolator=interp1d(k_have,P_have,bounds_error=avoid_extrapolation,fill_value="extrapolate")
        P_want=P_interpolator(k_want)
    return (k_want,P_want)

def extrapolation_warning(regime,want,have):
    print("WARNING: if extrapolation is permitted in interpolate_P call, it will be conducted for {:15s} (want {:9.4}, have{:9.4})".format(regime,want,have))
    return None

def flip(n,Nvox):
    """
    philosophy:
    if Nvox is even, send i=-Nvox//2 to -Nvox//2; for all other i, send i to -i

    inputs:
    n     = index you want to flip
    odd   = flag (1/0, not actually boolean) for the number of voxels per side of the box
    Nvox = number of voxels per box side

    outputs:
    index flipped according to the above philosophy
    """
    odd=Nvox%2
    if ((not odd) and (n==-Nvox//2)):
        pass
    else:
        n=-n
    return n   

def generate_box(P,k,Lsurvey,Nvox):
    """
    philosophy:
    generate a brightness temperature box consistent with a given matter power spectrum

    inputs:
    P = power spectrum
    k = wavenumber-space points at which the power spectrum is sampled
    Lsurvey = length of a cosmological box side, in Mpc (just sets the volume element scaling... nothing to do with pixelization)
    Nvox = number of voxels to create per cosmo box side

    outputs:
    Nvox x Nvox x Nvox brightness temp box
    """
    print("entering generate_box")
    t0=time.time()
    # helper variable setup
    k=k.real # enforce what makes sense physically
    P=P.real
    Npix=len(P)
    assert(Nvox>=Npix), "Nvox>=Npix is baked into the code at the moment. I'm going to fix this (interpolation...) after I handle the more pressing issues, but for now, why would you even want Nvox<Npix?"
    Delta = Lsurvey/Nvox # voxel side length
    dr3 = Delta**3 # voxel volume
    twopi = 2*np.pi
    V=Lsurvey**3
    r=twopi/k 
    print("generate_box: Lsurvey,Nvox,k,P=",Lsurvey,Nvox,k,P)
    
    # CORNER-origin r grid
    rmags=Lsurvey*np.fft.fftfreq(Nvox)
    RX,RY,RZ=np.meshgrid(rmags,rmags,rmags)
    rgrid=np.sqrt(RX**2+RY**2+RZ**2)
    print("generate_box: rgrid.shape=",rgrid.shape)
    
    # take appropriate draws from normal distributions to populate T-tilde
    sigmas=np.flip(np.sqrt(V*P/2)) # has Npix elements ... each element describes the T-tilde values in that k-bin ... flip to anticipate the fact that I'm working in r-space but calculated this vector in k-space
    sigmas=np.reshape(sigmas,(len(k),))
    print("generate_box: sigmas=",sigmas)
    nsigmas=len(sigmas)
    # nsigmas=np.max([len(sigmas),len(sigmas[0])])
    sigmas=np.reshape(sigmas,(nsigmas,)) # transition from the (1,npts) of the CAMB PS to (npts,) ... I think this became a problem in May because I got rid of some hard-coded reshaping in get_mps
    Ttre=np.zeros((Nvox,Nvox,Nvox))
    Ttim=np.zeros((Nvox,Nvox,Nvox))
    ##
    bin_indices=np.digitize(rgrid,r,right=False) # must pass x,bins; rgrid is the big box and r has floors
    for i,binedge in enumerate(r):
        sig=sigmas[i]
        here=np.nonzero(i==bin_indices) # all box indices where the corresp bin index is the ith binedge (iterable)
        numhere=len(np.argwhere(i==bin_indices)) # number of voxels in the bin we're currently considering
        sampsRe=np.random.normal(scale=sig, size=(numhere,)) # samples for filling the current bin
        sampsIm=np.random.normal(scale=sig, size=(numhere,))
        if (numhere>0):
            Ttre[here]=sampsRe
            Ttim[here]=sampsIm
    ##

    # ##### START OF MANUAL APPROXIMATION TO IRFFTN
    # # force the appropriate special cases to be real
    # h=  Nvox//2
    # H=h # upper lim of indexing -> if odd, overwrite (indexing needs to stop at N//2, not N//2-1, and for i in range(a,b) stops at b-1)
    # odd=Nvox%2
    # if(odd):
    #     H+=1
    # else:
    #     Ttim[-h,-h,-h]=0.
    #     Ttim[-h,-h, 0]=0.
    #     Ttim[-h, 0,-h]=0.
    #     Ttim[-h, 0, 0]=0.
    #     Ttim[ 0,-h,-h]=0.
    #     Ttim[ 0,-h, 0]=0.
    #     Ttim[ 0, 0,-h]=0.
    # Ttim[0, 0, 0]=0.
    # Tt=Ttre+1j*Ttim 

    # # apply the symmetries
    # for kx in range(-h,H): 
    #     kxf=flip(kx,Nvox)
    #     for ky in range(-h,H):
    #         kyf=flip(ky,Nvox)
    #         for kz in range(-h,H):
    #             kzf=flip(kz,Nvox)
    #             Tt[kx,ky,kz]=np.conj(Tt[kxf,kyf,kzf]) # !!still in corner-based indices
    
    # T=np.fft.ifftn(Tt) # numpy needs array indexing to be corner-based to take FFTs
    # T=(np.fft.fftshift(T)/dr3).real # send origin back to center (physics coordinates) and take T out of integral-land (it's real by this point [yes, I checked], but I do need to take the extra step of saving discarding the imag part [nonzero at roughly the machine precision level, O(1e-13)] to avoid future headaches)
    # ##### END OF MANUAL APPROXIMATION TO IRFFTN

    ##### START OF TRIAL CASE WHERE I USE IRFFTN TO AVOID HAVING TO APPLY THE SYMMETRIES MANUALLY
    Tt=Ttre+1j*Ttim # still no symmetries 
    T=np.fft.fftshift(np.fft.irfftn(Tt))/dr3
    ##### END OF TRIAL CASE WHERE I USE IRFFTN TO AVOID HAVING TO APPLY THE SYMMETRIES MANUALLY
    print("Nvox=",Nvox,"box generated in",time.time()-t0,"s")
    return rgrid,T,rmags