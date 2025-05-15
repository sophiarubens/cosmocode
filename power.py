import numpy as np

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

def P(T, k, Lsurvey, Npix):
    '''
    philosophy:
    * generate a power spectrum consistent with a brightness temperature box for which you already know the k-bins
    * in practice, to obtain a power spectrum you should access this routine through a wrapper:
        * ps_userbin(T, kbins, Lsurvey, Npix) -> if you need maximum binning flexibility (e.g. hybrid lin-log)
        * ps_autobin(T, mode, Lsurvey, Npix) -> if you need simple linear or logarithmic bins

    inputs:
    T       = Lsurvey-x-Lsurvey-x-Lsurvey data box for which you wish to create the power spectrum
    k       = k-bins for power spectrum indexing ... designed to be provided by one of two wrapper functions: user-created (custom) bins or automatically generated (linear- or log-spaced) bins
    Lsurvey = side length of the simulation cube (Mpc)
    Npix    = number of bins per side of the simulation cube

    outputs:
    Npix x 2 array storing kbin,P(kbin) ordered pairs
    '''
    # helper variables
    Delta = Lsurvey/Npix # voxel side length
    dr3 = Delta**3 # size of voxel
    twopi=2*np.pi
    
    # process the box values
    Ts = np.fft.ifftshift(T)*dr3 # T-ishifted (np wants a corner origin; ifftshift takes you there)
    Tts = np.fft.fftn(Ts) # T-tilde
    Tt = np.fft.fftshift(Tts) # shift back to physics land
    mTt = Tt*np.conjugate(Tt) # mod-squared of Tt
    mTt = mTt.real # I checked, and there aren't even any machine precision issues in the imag part
    
    # prepare to tie the processed box values to relevant k-values (generate k-quantities)
    Kshuf=twopi*np.fft.fftshift(np.fft.fftfreq(Lsurvey,d=Delta)) # fftshift doesn't care whether you're dealing with correlations, brightness temps, or config/Fourier space coordinates ... the math is the same!
    KX,KY,KZ=np.meshgrid(Kshuf,Kshuf,Kshuf)
    kmags=np.sqrt(KX**2+KY**2+KZ**2)
    binidxs=np.digitize(kmags,k,right=False)

    # identify which box each point falls into
    amTt=np.zeros(Npix)
    for i,binedge in enumerate(k):
        here=np.nonzero(i==binidxs) # want all box indices where the corresp bin index is the ith binedge (nonzero is like argwhere, but gives output that is suitable for indexing arrays)
        if (len(here[0])>0): # already know len(here) = arr_dimns, so check for emptiness w/ len(here[0])
            amTt[i]=np.mean(mTt[here])
        else:
            amTt[0]=0

    V = Lsurvey**3 # volume of the simulation cube
    return [np.array(k),np.array(amTt/V)]

def ps_userbin(T, kbins, Lsurvey, Npix):
    '''
    philosophy: 
    * generate a power spectrum with user-provided bins, consistent with a given brightness temperature box
    * wrapper function for the power spectrum function P(T, k, Lsurvey, Npix) 
    * use when you need more bin customization flexibility than mere lin- or log-spacing

    inputs:
    T       = Lsurvey-x-Lsurvey-x-Lsurvey data box for which you wish to create the power spectrum
    kbins   = user-defined (if using this wrapper function, then probably custom-spaced) VECTOR of k-bins
    Lsurvey = side length of the simulation cube (Mpc)
    Npix    = number of **bins** per side of the simulation cube

    outputs:
    Npix x 2 array storing kbin,P(kbin) ordered pairs
    '''  
    kmax=kbins[-1]
    kmin=kbins[0]
    if (kmax<kmin):
        kbins=np.flip(kbins)
        kmax=kbins[-1]
        kmin=kbins[0]
    elif (kmax==kmin):
        assert(1==0), "kbins is either constant or non-monotonic -> cannot create useful bins from these" # force quit ... the always false condition is just a hanger for the warning message
    twopi=2*np.pi
    Delta=Lsurvey/Npix
    assert (kmax<=twopi/Delta and kmin>=twopi/Lsurvey), 'the provided k-range must be a subset of the BCS range probed by this survey'
    return P(T,kbins,Lsurvey,Npix)

def ps_autobin(T, mode, Lsurvey, Npix):
    '''
    philosophy:
    * generate a power spectrum with lin- or log-spaced bins, consistent with a given brightness temperature box
    * wrapper function for the power spectrum function P(T, k, Lsurvey, Npix) 

    inputs:
    T       = Lsurvey-x-Lsurvey-x-Lsurvey data box for which you wish to create the power spectrum
    mode    = binning mode (linear or logarithmic)
    Lsurvey = side length of the simulation cube (Mpc)
    Npix    = number of **bins** per side of the simulation cube

    outputs:
    Npix x 2 array storing kbin,P(kbin) ordered pairs
    '''
    twopi=2*np.pi
    Delta=Lsurvey/Npix
    kbins=np.logspace(np.log10(twopi/Lsurvey),np.log10(twopi/Delta),num=Npix)
    assert (mode=='lin' or mode=='log'), 'only linear and logarithmic auto-generated bins are currently supported'
    kmin=twopi/Npix
    kmax=twopi/Lsurvey
    if (mode=='log'):
        kbins=np.logspace(np.log10(kmin),np.log10(kmax),num=Npix)
    else:
        kbins=np.linspace(kmin,kmax,Npix)
    return P(T,kbins,Lsurvey,Npix)

def flip(n,nfvox):
    '''
    philosophy:
    if nfvox is even, send i=-nfvox//2 to -nfvox//2; for all other i, send i to -i

    inputs:
    n     = index you want to flip
    odd   = flag (1/0, not actually boolean) for the number of voxels per side of the cube
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
    Lsurvey = length of a simulation cube side, in Mpc
    nfvox = number of voxels to create per sim cube side

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
    Ttre=np.zeros((nfvox,nfvox,nfvox))
    Ttim=np.zeros((nfvox,nfvox,nfvox))
    binidxs=np.digitize(rgrid,r,right=False) # must pass x,bins; rgrid is the big box and r has floors
    for i,binedge in enumerate(r):
        sig=sigmas[i]
        here=np.nonzero(i==binidxs) # all box indices where the corresp bin index is the ith binedge (iterable)
        numhere=len(np.argwhere(i==binidxs)) # number of voxels in the bin we're currently considering
        print("in ips: sig,numhere=",sig,numhere)
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
    
    # return rgrid,T # in place before May 15th
    return rgrid,T,rmags