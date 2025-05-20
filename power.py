import numpy as np
from matplotlib import pyplot as plt
import time

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

def P(T, k, Lsurvey, binto="sph"):
    '''
    philosophy:
    * generate a power spectrum consistent with a brightness temperature box for which you already know the k-bins
    * in practice, to obtain a power spectrum you should access this routine through a wrapper:
        * ps_userbin(T, kbins, Lsurvey) -> if you need maximum binning flexibility (e.g. hybrid lin-log)
        * ps_autobin(T, mode, Lsurvey) -> if you need simple linear or logarithmic bins

    inputs:
    T       = Npix x Npix x Npix data box for which you wish to create the power spectrum
    k       = k-bins for power spectrum indexing ... designed to be provided by one of two wrapper functions: user-created (custom) bins or automatically generated (linear- or log-spaced) bins
                 - if binto=="sph": k is assumed to have shape (Nk,) or similar (e.g. (Nk,1) or (1,Nk))
                 - if binto=="cyl": k is assumed to be a (possibly ragged) array that can be unpacked as kpar,kperp=k where kpar has shape (Nkpar,) (or similar) and kperp has shape (Nkperp,) or similar
    Lsurvey = side length of the simulation cube (Mpc) (just sets the volume element scaling... nothing to do with pixelization)
    binto   = how to bin the resulting power spectrum (current options: spherically (1D) or cylindrically (2D, flat-sky approximation))

    outputs:
    k-modes and powers of the power spectrum
         - if binto=="sph": [(Nk,),(Nk,)] object unpackable as kbins,powers
         - if binto=="cyl": [(Nkpar,Nkperp),(Nkpar,Nkperp),(Nkpar,),(Nkperp,)] object unpackable as kgrid,powers,kparvec,kperpvec
    '''
    # helper variables
    # print("T.shape=",T.shape)
    Npix=T.shape[0]
    Delta = Lsurvey/Npix # voxel side length
    # print("P: Lsurvey,Npix,Delta=",Lsurvey,Npix,Delta)
    dr3 = Delta**3 # size of voxel
    twopi=2*np.pi
    V = Lsurvey**3 # volume of the simulation cube
    
    # process the box values
    Ts = np.fft.ifftshift(T)*dr3 # T-ishifted (np wants a corner origin; ifftshift takes you there)
    Tts = np.fft.fftn(Ts) # T-tilde
    Tt = np.fft.fftshift(Tts) # shift back to physics land
    mTt = Tt*np.conjugate(Tt) # mod-squared of Tt
    mTt = mTt.real # I checked, and there aren't even any machine precision issues in the imag part

    # establish Cartesian Fourier duals to box coordinates
    Kshuf=twopi*np.fft.fftshift(np.fft.fftfreq(Npix,d=Delta)) # fftshift doesn't care whether you're dealing with correlations, brightness temps, or config/Fourier space coordinates ... the math is the same!
    KX,KY,KZ=np.meshgrid(Kshuf,Kshuf,Kshuf)
    ## PROBABLY NEED TO ADD AN INDEXING KEYWORD WHICH BECOMES IMPORTANT FOR THE CASE OF CYLINDRICAL SYMMETRY BECAUSE THE INDICES ARE NO LONGER ALL INTERCHANGEABLE
    
    if (binto=="sph"):
        Nk=len(k)

        # prepare to tie the processed box values to relevant k-values (prep for binning)
        kmags=np.sqrt(KX**2+KY**2+KZ**2) # BINNING SPHERICALLY literally just the Fourier duals to the config space points ... not the binned k-values yet
        binidxs=np.digitize(kmags,k,right=False)
        binidxs_1d=np.reshape(binidxs,(Npix**3,))
        mTt_1d=    np.reshape(mTt,    (Npix**3,))

        # binning
        summTt=np.bincount(binidxs_1d,weights=mTt_1d,minlength=Nk)
        NmTt=  np.bincount(binidxs_1d,               minlength=Nk)
    elif (binto=="cyl"):
        # print("in the binto=='cyl' branch of P: ")
        kpar,kperp=k # this assumes my apparently-nontraditional convention of putting kpar first... fix this later, probably
        Nkpar=len(kpar)
        Nkperp=len(kperp)
        # print("Nkpar,Nkperp=",Nkpar,Nkperp)

        # prepare to tie the processed box values to relevant k-values (prep for binning)
        kparmags=KZ
        kperpmags=np.sqrt(KX**2+KY**2)
        parbinidxs= np.digitize(kparmags, kpar, right=False)
        perpbinidxs=np.digitize(kperpmags,kperp,right=False)
        parbinidxs_1d= np.reshape(parbinidxs, (Npix**3,))
        perpbinidxs_1d=np.reshape(perpbinidxs,(Npix**3,))
        print("parbinidxs.min(),parbinidxs.max()=",parbinidxs.min(),parbinidxs.max())
        print("perpbinidxs.min(),perpbinidxs.max()=",perpbinidxs.min(),perpbinidxs.max())

        # binning

        # translation to power spectrum terms
        amTt=np.zeros(Nk)
        nonemptybins=np.non

        # identify which point each box falls into
        amTt=np.zeros((Nkpar,Nkperp))
        for i,parbinedge in enumerate(kpar):
            for j,perpbinedge in enumerate(kperp):
                here=np.nonzero(np.logical_and(i==parbinidxs,j==perpbinidxs)) # need to verify this generalization to 2D
                if (len(here[0])>0):
                    amTt[i,j]=np.mean(mTt[here])
        kpargrid,kperpgrid=np.meshgrid(kpar,kperp,indexing="ij") # need to verify that the indexing actually does what I mean for it to do
        P=np.array(amTt/V)
        # print("before return: kpargrid.shape,kperpgrid.shape,P.shape,kpar.shape,kperp.shape=",kpargrid.shape,kperpgrid.shape,P.shape,kpar.shape,kperp.shape)
        return [[kpargrid,kperpgrid],P,kpar,kperp]
    else:
        assert(1==0), "only spherical and cylindrical power spectrum binning are currently supported"
        return None
    
    # translation to power spectrum terms
    amTt=np.zeros(Nk)
    nonemptybins=np.nonzero(NmTt)
    amTt[nonemptybins]=summTt[nonemptybins]/NmTt[nonemptybins]
    # t1=time.time()
    # print("P: binning took",t1-t0,"s")
    P=np.array(amTt/V)
    return [k,P]

def ps_userbin(T, kbins, Lsurvey, binto="sph"):
    '''
    philosophy: 
    * generate a power spectrum with user-provided bins, consistent with a given brightness temperature box
    * wrapper function for the power spectrum function P(T, k, Lsurvey, Npix) 
    * use when you need more bin customization flexibility than mere lin- or log-spacing

    inputs:
    T       = Npix x Npix x Npix data box for which you wish to create the power spectrum
    kbins   = user-defined (if using this wrapper function, then probably custom-spaced) k-bins
                - if binto=="sph": vector of shape (Nk,) or similar
                - if binto=="cyl": potentially ragged array [(Nkpar,),(Nkperp,)] or similar, unpackable as kpar,kperp=kbins
    Lsurvey = side length of the simulation cube (Mpc) (just sets the volume element scaling... nothing to do with pixelization)

    outputs:
    one copy of P() output
    '''  
    Npix = T.shape[0] # assumes that you use the function as intended and pass a data cube
    if (binto=="sph"):
        kmax=kbins[-1]
        kmin=kbins[0]
        if (kmax<kmin):
            kbins=np.flip(kbins)
            # kmax=kbins[-1] # not actually used anywhere ... leave commented for now to make sure things behave as expected/ that I'm not missing anything before actually removing
            # kmin=kbins[0]
        elif (kmax==kmin):
            assert(1==0), "constant or non-monotonic k bins -> try again with more sensible binning" # force quit ... the always false condition is just a hanger for the warning message
        twopi=2*np.pi
        Delta=Lsurvey/Npix
        Nk=len(kbins)
    elif (binto=="cyl"):
        kpar,kperp=kbins
        kparmin,kparmax=kpar[0],kpar[-1]
        kperpmin,kperpmax=kperp[0],kperp[-1]
        if (kparmax<kparmin):
            kpar=np.flip(kbins)
        elif (kparmax==kparmin):
            assert(1==0), "constant or non-monotonic kpar bins -> try again with more sensible binning"
        if (kperpmax<kperpmin):
            kperp=np.flip(kperp)
        elif(kperpmax==kperpmin):
            assert(1==0), "constant or non-monotonic kperp bins -> try again with more sensible binning"
        kbins=[kpar,kperp]
    else:
        assert(1==0), "only spherical and cylindrical power spectrum binning are currently supported"

    return P(T,kbins,Lsurvey, binto=binto)

def ps_autobin(T, mode, Lsurvey, Nk):
    '''
    philosophy:
    * generate a spherically binned power spectrum with lin- or log-spaced bins, consistent with a given brightness temperature box
    * wrapper function for the power spectrum function P(T, k, Lsurvey, Npix) 
    * I could generalize this to calculate cylindrical bins accessed by a particular instrument, but I'd need to pass a slew of survey parameters to this function, and for now it seems cleaner to precalculate them the way you want and then pass to the custom bin wrapper

    inputs:
    T       = Npix x Npix x Npix data box for which you wish to create the power spectrum
    mode    = binning mode (linear or logarithmic)
    Lsurvey = side length of the simulation cube (Mpc) (just sets the volume element scaling... nothing to do with pixelization)
    Nk      = number of k-bins to include in the power spectrum

    outputs:
    one copy of P() output
    '''
    Npix=T.shape[0]
    twopi=2*np.pi
    Delta=Lsurvey/Npix
    assert (mode=='lin' or mode=='log'), 'only linear and logarithmic auto-generated bins are currently supported'
    
    kmax=twopi/Delta # was Npix but I don't think that makes sense anymore
    kmin=twopi/Lsurvey
    if (mode=='log'):
        # kbins=np.logspace(np.log10(kmin),np.log10(kmax),num=Nk+1)[:-1] # bypass the poor stats for the largest k-bin
        kbins=np.logspace(np.log10(kmin),np.log10(kmax),num=Nk)
    else:
        # kbins=np.linspace(kmin,kmax,Nk+1)[:-1]
        kbins=np.linspace(kmin,kmax,Nk)
    return P(T,kbins,Lsurvey)

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
    Lsurvey = length of a simulation cube side, in Mpc (just sets the volume element scaling... nothing to do with pixelization)
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
    # print("ips: Lsurvey,nfvox=",Lsurvey,nfvox)
    rmags=Lsurvey*np.fft.fftfreq(nfvox)
    # print("ips: rmags=",rmags)
    RX,RY,RZ=np.meshgrid(rmags,rmags,rmags)
    rgrid=np.sqrt(RX**2+RY**2+RZ**2)
    
    # take appropriate draws from normal distributions to populate T-tilde
    sigmas=np.flip(np.sqrt(V*P/2)) # has Npix elements ... each element describes the T-tilde values in that k-bin ... flip to anticipate the fact that I'm working in r-space but calculated this vector in k-space
    sigmas=np.reshape(sigmas,(sigmas.shape[1],)) # transition from the (1,npts) of the CAMB PS to (npts,) ... I think this became a problem in May because I got rid of some hard-coded reshaping in get_mps
    Ttre=np.zeros((nfvox,nfvox,nfvox))
    Ttim=np.zeros((nfvox,nfvox,nfvox))
    binidxs=np.digitize(rgrid,r,right=False) # must pass x,bins; rgrid is the big box and r has floors
    for i,binedge in enumerate(r):
        sig=sigmas[i]
        here=np.nonzero(i==binidxs) # all box indices where the corresp bin index is the ith binedge (iterable)
        numhere=len(np.argwhere(i==binidxs)) # number of voxels in the bin we're currently considering
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
Npix = 99
Nk = 12

# kcustom=np.logspace(np.log10(2.*np.pi/100),np.log10(5),Nk)
plt.figure()
maxvals=0.0
for i in range(10):
    T = np.random.normal(loc=0.0, scale=1.0, size=(Npix,Npix,Npix))
    # kfloors,vals=ps_userbin(T,kcustom,Lsurvey)
    kfloors,vals=ps_autobin(T,"log",Lsurvey,Nk)
    plt.scatter(np.log(kfloors),vals)
    maxvalshere=np.max(vals)
    if (maxvalshere>maxvals):
        maxvals=maxvalshere
plt.xlabel("log(k*1Mpc)")
plt.ylabel("Power (K$^2$ Mpc$^3$)")
plt.title("Test PS calc for Lsurvey,Npix,Nk={:4},{:4},{:4}".format(Lsurvey,Npix,Nk))
plt.ylim(0,1.2*maxvals)
plt.show()

# ############## TEST CYL FWD
# Lsurvey = 8
# # Nkpar = 327
# # Nkperp = 1010
# Nkpar=12
# Nkperp=7

# kperp=np.linspace(0.09,3.3,Nkperp)
# kpar=np.linspace(0.02,6.1,Nkpar)
# deltakpar=kpar[1]-kpar[0]
# deltakperp=kperp[1]-kperp[0]
# largerdeltak=np.max((deltakpar,deltakperp))

# Npix = 99 # bad bc things get skipped
# kcustom=[kpar,kperp]
# nsubrow=3
# nsubcol=3
# vmin=np.infty
# vmax=-np.infty
# fig,axs=plt.subplots(nsubrow,nsubcol)
# for i in range(nsubrow):
#     for j in range(nsubcol):
#         T = np.random.normal(loc=0.0, scale=1.0, size=(Npix,Npix,Npix))
#         [kpargrid_returned,kperpgrid_returned],vals,kpar_returned,kperp_returned=ps_userbin(T,kcustom,Lsurvey,binto="cyl")
#         im=axs[i,j].imshow(vals,origin="lower",extent=[kpar[0],kpar[-1],kperp[0],kperp[-1]])
#         axs[i,j].set_ylabel("$k_{||}$")
#         axs[i,j].set_xlabel("$k_\perp$")
#         minval=np.min(vals)
#         maxval=np.max(vals)
#         if (minval<vmin):
#             vmin=minval
#         if (maxval>vmax):
#             vmax=maxval
# fig.colorbar(im,extend="both")
# plt.suptitle("Test cyl PS calc")
# plt.tight_layout()
# plt.show()

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