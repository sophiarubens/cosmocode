import numpy as np
from matplotlib import pyplot as plt
import time

pi=np.pi
twopi=2.*pi

'''
this module helps connect ensemble-averaged power spectrum estimates and cosmological brighness temperature boxes for two main use cases:
1. // forward direction: generate a power spectrum that describes the statistics of a brightness temperature box
2. // backwards direction: generate one realization of a brightness temperature box consistent with a known power spectrum

call structure:
1. ps_userbin
   ps_autobin -> {{P}}
2. ips -> {{flip}}

{{}} = generally no need to call directly; called as necessary by higher-level functions
'''

def P(T, k, Lsurvey):
    '''
    philosophy:
    * generate a power spectrum consistent with a brightness temperature box for which you already know the k-bins
    * in practice, to obtain a power spectrum you should access this routine through a wrapper:
        * ps_userbin(T, kbins, Lsurvey) -> if you need maximum binning flexibility (e.g. hybrid lin-log)
        * ps_autobin(T, mode, Lsurvey) -> if you need simple linear or logarithmic bins

    inputs:
    T       = Npix x Npix x Npix data box for which you wish to create the power spectrum
    k       = k-bins for power spectrum indexing ... designed to be provided by one of two wrapper functions: user-created (custom) bins or automatically generated (linear- or log-spaced) bins
                 - shape (Nk,) or similar (e.g. (Nk,1) or (1,Nk)) for spherical binning
                 - shape [(Nkpar,),(Nkperp,)] or similar for cylindrical binning
    Lsurvey = side length of the cosmological box (Mpc) (just sets the volume element scaling... nothing to do with pixelization)

    outputs:
    k-modes and powers of the power spectrum
         - if binto=="sph": [(Nk,),(Nk,)] object unpackable as kbins,powers
         - if binto=="cyl": [[(Nkpar,),(Nkperp,)],(Nkpar,Nkperp)] object unpackable as [kparvec,kperpvec],powers_on_grid **MAKE SURE THIS ENDS UP BEING TRUE**
    '''
    # helper variables
    lenk=len(k)
    if lenk==2:
        binto="cyl"
    else: # kind of hackily relying on the properties of list-stored ragged arrays to let the "if" catch cases that need to be cyl and then shuffle everything else along to spherical ... not very robust against pathological calls
        binto="sph"
    Npix  =T.shape[0]
    Delta = Lsurvey/Npix # voxel side length
    dr3   = Delta**3     # voxel volume
    V     = Lsurvey**3   # volume of the cosmo box
    
    # process the box values
    Ts  = np.fft.ifftshift(T)*dr3 # T-ishifted (np wants a corner origin; ifftshift takes you there)
    Tts = np.fft.fftn(Ts)         # T-tilde
    Tt  = np.fft.fftshift(Tts)    # shift back to physics land
    mTt = Tt*np.conjugate(Tt)     # mod-squared of Tt
    mTt = mTt.real                # I checked, and there aren't even any machine precision issues in the imag part

    # establish Cartesian Fourier duals to box coordinates
    k_vec_for_box= twopi*np.fft.fftshift(np.fft.fftfreq(Npix,d=Delta))
    kx_box_grid,ky_box_grid,kz_box_grid=      np.meshgrid(k_vec_for_box,k_vec_for_box,k_vec_for_box,indexing="ij") # centre-origin Fourier duals to config space coords (ofc !not !yet !binned)

    if (binto=="sph"):
        Nk=lenk # number of k-modes to put in the power spectrum

        # prepare to tie the processed box values to relevant k-values
        k_box=          np.sqrt(kx_box_grid**2+ky_box_grid**2+kz_box_grid**2) # scalar k for each voxel
        bin_indices=    np.digitize(k_box,k,right=False)                      # box with entries indexing which bin each voxel belongs in
        bin_indices_1d= np.reshape(bin_indices,(Npix**3,))                    # to bin, I use np.bincount, which requires 1D input
        mTt_1d=         np.reshape(mTt,    (Npix**3,))                        # ^ same preprocessing

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
        perpbin_indices_slice_1d= np.reshape(perpbin_indices_slice,(Npix**2,))   # even though I've chosen a representative slice, I still need to flatten down to 1D in anticipation of bincounting
        parbin_indices_column=    np.digitize(k_vec_for_box,kpar, right=False)   # vector with entries indexing which kpar bin each voxel belongs in (pending slight postprocessing in the loop) ... just as I could look at a representative slice for the kperp direction, I can look at a representative chunk for the LoS direction (though, naturally, in this case it is a "column") ... no need to reshape, b/c (1). it's already 1D and (2). I don't have an explicit bincount call along this axis because I iterate over kpar slices

        # binning 
        summTt= np.zeros((Nkpar,Nkperp)) # for the ensemble average: sum    of mTt values in each bin  ... each time I access it, I'll access the kparBIN row of interest, but update all NkperpBIN columns
        NmTt=   np.zeros((Nkpar,Nkperp)) # for the ensemble average: number of mTt values in each bin
        for i in range(Npix): # iterate over the kpar axis of the box to capture all LoS slices
            if (i==0): # stats of the kperp "bull's eye" slice
                slice_bin_counts= np.bincount(perpbin_indices_slice_1d, minlength=Nkperp) # each slice's update to the denominator of the ensemble average
                if (len(slice_bin_counts)==(Nkperp+1)): # prune central voxel "below the floor of the lowest bin" extension bin
                    slice_bin_counts=slice_bin_counts[1:]
            mTt_slice=       mTt[:,:,i]                                                                  # take the slice of interest of the preprocessed box values
            mTt_slice_1d=    np.reshape(mTt_slice,(Npix**2,))                                            # reshape to 1D for bincount compatibility
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

def get_bins(Npix,Lsurvey,Nk,mode):
    Delta=Lsurvey/Npix
    twopi=2.*np.pi
    kmax=twopi/Delta
    kmin=twopi/Lsurvey
    if (mode=="log"):
        kbins=np.logspace(np.log10(kmin),np.log10(kmax),num=Nk)
        arg=np.log(Npix)/Nk
        limiting_spacing=twopi*(10.**(2.*arg)-10.**arg)
    elif (mode=="lin"):
        kbins=np.linspace(kmin,kmax,Nk)
        limiting_spacing=twopi*(Npix-1)/(Nk*Lsurvey)
    else:
        assert(1==0), "only log and linear binning are currently supported"
    return kbins,limiting_spacing

class ResolutionError(Exception):
    pass

def ps_autobin(T, mode, Lsurvey, Nk0, Nk1=0):
    '''
    philosophy:
    * generate a spherically binned power spectrum with lin- or log-spaced bins, consistent with a given brightness temperature box
    * wrapper function for the power spectrum function P(T, k, Lsurvey, Npix) 
    * I could generalize this to calculate cylindrical bins accessed by a particular instrument, but I'd need to pass a slew of survey parameters to this function, and for now it seems cleaner to precalculate them the way you want and then pass to the custom bin wrapper

    inputs:
    T       = Npix x Npix x Npix data box for which you wish to create the power spectrum
    mode    = binning mode (linear or logarithmic)
    Lsurvey = side length of the cosmological box (Mpc) (just sets the volume element scaling... nothing to do with pixelization)
    Nk0      = number of k-bins to include in the power spectrum (if this is the only nonzero Nk, the power spectrum will be binned spherically)
    Nk1     = number of k-bins to include along axis=1 of the power spectrum (if nonzero, the power spectrum will be binned cylindrically)

    outputs:
    one copy of P() output
    '''
    Npix=T.shape[0]
    deltak_box=twopi/Lsurvey

    k0bins,limiting_spacing_0=get_bins(Lsurvey,Npix,Nk0,mode)
    if (limiting_spacing_0<deltak_box):
        raise ResolutionError
    
    if (Nk1>0):
        k1bins,limiting_spacing_1=get_bins(Lsurvey,Npix,Nk1,mode)
        if (limiting_spacing_1<deltak_box):
            raise ResolutionError
        kbins=[k0bins,k1bins]
    else:
        kbins=k0bins
    
    return P(T,kbins,Lsurvey)

def flip(n,nfvox):
    '''
    philosophy:
    if nfvox is even, send i=-nfvox//2 to -nfvox//2; for all other i, send i to -i

    inputs:
    n     = index you want to flip
    odd   = flag (1/0, not actually boolean) for the number of voxels per side of the box
    nfvox = number of voxels per box side

    outputs:
    index flipped according to the above philosophy
    '''
    odd=nfvox%2
    if ((not odd) and (n==-nfvox//2)):
        pass
    else:
        n=-n
    return n   

def ips(P,k,Lsurvey,nfvox):
    '''
    philosophy:
    generate a brightness temperature box consistent with a given matter power spectrum

    inputs:
    P = power spectrum
    k = wavenumber-space points at which the power spectrum is sampled
    Lsurvey = length of a cosmological box side, in Mpc (just sets the volume element scaling... nothing to do with pixelization)
    nfvox = number of voxels to create per cosmo box side

    outputs:
    nfvox x nfvox x nfvox brightness temp box
    '''
    # helper variable setup
    k=k.real # enforce what makes sense physically
    P=P.real
    Npix=len(P)
    assert(nfvox>=Npix), "nfvox>=Npix is baked into the code at the moment. I'm going to fix this (interpolation...) after I handle the more pressing issues, but for now, why would you even want nfvox<Npix?"
    Delta = Lsurvey/nfvox # voxel side length
    dr3 = Delta**3 # voxel volume
    twopi = 2*np.pi
    V=Lsurvey**3
    r=twopi/k # gives the right thing but not 100% sure why it breaks if I add back the /Lsurvey I thought belonged
    
    # CORNER-origin r grid    
    rmags=Lsurvey*np.fft.fftfreq(nfvox)
    RX,RY,RZ=np.meshgrid(rmags,rmags,rmags)
    rgrid=np.sqrt(RX**2+RY**2+RZ**2)
    
    # take appropriate draws from normal distributions to populate T-tilde
    sigmas=np.flip(np.sqrt(V*P/2)) # has Npix elements ... each element describes the T-tilde values in that k-bin ... flip to anticipate the fact that I'm working in r-space but calculated this vector in k-space
    sigmas=np.reshape(sigmas,(sigmas.shape[1],)) # transition from the (1,npts) of the CAMB PS to (npts,) ... I think this became a problem in May because I got rid of some hard-coded reshaping in get_mps
    Ttre=np.zeros((nfvox,nfvox,nfvox))
    Ttim=np.zeros((nfvox,nfvox,nfvox))
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

    # force the appropriate special cases to be real
    h=  nfvox//2
    H=h # upper lim of indexing -> if odd, overwrite (indexing needs to stop at N//2, not N//2-1, and for i in range(a,b) stops at b-1)
    odd=nfvox%2
    if(odd):
        H+=1
    else:
        Ttim[-h,-h,-h]=0.
        Ttim[-h,-h, 0]=0.
        Ttim[-h, 0,-h]=0.
        Ttim[-h, 0, 0]=0.
        Ttim[ 0,-h,-h]=0.
        Ttim[ 0,-h, 0]=0.
        Ttim[ 0, 0,-h]=0.
    Ttim[0, 0, 0]=0.
    Tt=Ttre+1j*Ttim 

    # apply the symmetries
    for kx in range(-h,H): 
        kxf=flip(kx,nfvox)
        for ky in range(-h,H):
            kyf=flip(ky,nfvox)
            for kz in range(-h,H):
                kzf=flip(kz,nfvox)
                Tt[kx,ky,kz]=np.conj(Tt[kxf,kyf,kzf]) # !!still in corner-based indices
    
    T=np.fft.ifftn(Tt) # numpy needs array indexing to be corner-based to take FFTs
    T=(np.fft.fftshift(T)/dr3).real # send origin back to center (physics coordinates) and take T out of integral-land (it's real by this point [yes, I checked], but I do need to take the extra step of saving discarding the imag part [nonzero at roughly the machine precision level, O(1e-13)] to avoid future headaches)
    
    return rgrid,T,rmags

############## TEST SPH FWD
Lsurvey = 103
Npix = 200 # 150 looks ok for spherical (if a little stripy for cylindrical), but turning up to 200 means the lowest-k bin is always empty (for spherical and along both axes for cylindrical ... I think it's just b/c the log-spaced bins are so close together)
Nk = 14

plt.figure()
maxvals=0.0
for i in range(5):
    T = np.random.normal(loc=0.0, scale=1.0, size=(Npix,Npix,Npix))
    kfloors,vals=ps_autobin(T,"lin",Lsurvey,Nk)
    plt.scatter(kfloors,vals)
    maxvalshere=np.max(vals)
    if (maxvalshere>maxvals):
        maxvals=maxvalshere
    # break # just look at one iteration for now to iron out the empty bin issue
plt.xlabel("k (1/Mpc)")
plt.ylabel("Power (K$^2$ Mpc$^3$)")
plt.title("Test white noise P(k) calc for Lsurvey,Npix,Nk={:4},{:4},{:4}".format(Lsurvey,Npix,Nk))
plt.ylim(0,1.2*maxvals)
plt.savefig("wn_sph.png",dpi=500)
plt.show() # WORKS AS OF 14:28 20.05.25

# assert(1==0), "fix sph first"
# ############## TEST CYL FWD
Nkpar=9 # 327
Nkperp=12 # 1010
# Nkpar=300
# Nkperp=100

nsubrow=3
nsubcol=3
vmin=np.infty
vmax=-np.infty
fig,axs=plt.subplots(nsubrow,nsubcol,figsize=(8,10))
for i in range(nsubrow):
    for j in range(nsubcol):
        T = np.random.normal(loc=0.0, scale=1.0, size=(Npix,Npix,Npix))
        k,vals=ps_autobin(T,"lin",Lsurvey,Nkpar,Nk1=Nkperp) 
        kpar,kperp=k
        kpargrid,kperpgrid=np.meshgrid(kpar,kperp,indexing="ij")
        im=axs[i,j].pcolor(kpargrid,kperpgrid,vals)
        axs[i,j].set_ylabel("$k_{||}$")
        axs[i,j].set_xlabel("$k_\perp$")
        axs[i,j].set_title("Realization {:2}".format(i*nsubrow+j))
        axs[i,j].set_aspect("equal")
        minval=np.min(vals)
        maxval=np.max(vals)
        if (minval<vmin):
            vmin=minval
        if (maxval>vmax):
            vmax=maxval
fig.colorbar(im,extend="both")
plt.suptitle("Test white noise P(kpar,kperp) calc for Lsurvey,Npix,Nkpar,Nkperp={:4},{:4},{:4},{:4}".format(Lsurvey,Npix,Nkpar,Nkperp))
plt.tight_layout()
plt.savefig("wn_cyl.png",dpi=500)
plt.show()

############## TESTS BWD
# Lsurvey=100 # Mpc
# plot=True
# cases=['ps_wn_2px.txt','z5spec.txt','ps_wn_20px.txt']
# ncases=len(cases)
# if plot:
#     fig,axs=plt.subplots(2*ncases,3, figsize=(15,10)) # (3 power specs * 2 voxel schemes per power spec) = 6 generated boxes to look at slices of
# t0=time.time()
# for k,case in enumerate(cases):
#     kfl,P=np.genfromtxt(case,dtype='complex').T
#     Npix=len(P)

#     # n_field_voxel_cases=[4,3]
#     n_field_voxel_cases=[21,22]
#     # n_field_voxel_cases=[44,45] # 15 s
#     # n_field_voxel_cases=[65,66] # 24 s
#     # n_field_voxel_cases=[88,89] # 36 s
#     # n_field_voxel_cases=[99,100] # 97 s
#     for j,n_field_voxels in enumerate(n_field_voxel_cases):
#         tests=[0,n_field_voxels//2,n_field_voxels-3]
#         rgen,Tgen=ips(P,kfl,Lsurvey,n_field_voxels)
#         print('done with inversion for k,j=',k,j)
#         if plot:
#             for i,test in enumerate(tests):

#                 if len(cases)>1:
#                     im=axs[2*k+j,i].imshow(Tgen[:,:,test])
#                     fig.colorbar(im)
#                     axs[2*k+j,i].set_title('slice '+str(test)+'/'+str(n_field_voxels)+'; original box = '+str(case))
#                 else:
#                     im=axs[2*k+j,i].imshow(Tgen[:,:,test])
#                     fig.colorbar(im)
#                     axs[2*k+j,i].set_title('slice '+str(test)+'/'+str(n_field_voxels)+'; original box = '+str(case))

# if plot:
#     plt.suptitle('brightness temp box slices generated from inverting a PS I calculated')
#     plt.tight_layout()
#     plt.show()
#     fig.savefig('ips_tests.png')
#     t1=time.time()

# print('test suite took',t1-t0,'s')