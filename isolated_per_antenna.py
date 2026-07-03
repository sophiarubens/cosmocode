import numpy as np
from matplotlib import pyplot as plt
from scipy.fft import fftn,irfftn,fftshift,ifftshift,fftfreq
from forecasting_pipeline import beam_effects, comoving_distance, beam_type_distribution, get_padding, PA_Gaussian, Blackman_Harris_safe_for_FFT
from astropy import units as u
from scipy.interpolate import RectBivariateSpline as RBS
from scipy.signal import convolve
from astropy.cosmology import Planck18
from astropy import constants as const

# cosmological. all are Planck18, whether they come from astropy or not
H0=Planck18.H0
h=H0/100
Omegam=Planck18.Om0
Omegamh2=Omegam*h**2
Omegab=Planck18.Ob0
Omegabh2=Omegab*h**2
Omegach2=0.12011
OmegaLambda=0.6842
Omegak=0
Omegar=1-OmegaLambda-Omegam-Omegak
ln1010AS=3.0448
AS=np.exp(ln1010AS)/10**10
ns=0.96605
w=-1
Omegamh2=Omegam*h**2
pars_fidu=    [ H0,    Omegabh2,      Omegamh2,      AS,           ns,    w] # suitable for getting matter power spec
parnames_fidu=["H_0", "Omega_b h^2", "Omega_c h^2", "10^9 * A_S", "n_s", "w"]

pars_forecast=    [ H0,    Omegabh2,      Omegach2,      w  ] # expect a 21-cm experiment to provide insight into these
parnames_forecast=["H_0", "Omega_b h^2", "Omega_c h^2", "w"]

dpar_default=1e-3*np.ones(len(pars_fidu))
dpar_default[3]*=1e-9

# physical
nu_HI_z0=1420.405751768*u.MHz
c=const.c
dif_lim_prefac=1.029

# mathematical
pi=np.pi
twopi=2.*pi
ln2=np.log(2)

# numerical
maxint=   np.iinfo(np.int64  ).max
BasicAiryHWHM=1.616339948310703178119139753683896309743121097215461023581 # intentionally preposterous number of sig figs from Mathematica
eps=1e-15
dpi_to_use=250

# CHORD
N_NS_full=24
N_EW_full=22
b_NS=8.5*u.m
b_EW=6.3*u.m
b_max_CHORD=np.sqrt((N_NS_full*b_NS)**2+(N_EW_full*b_EW)**2)*u.m
DRAO_lat=49.320791*pi/180.*u.rad # Google Maps satellite view, eyeballing what looks like the middle of the CHORD site: 49.320791, -119.621842 (bc considering drift-scan CHIME-like "pointing at zenith" mode, same as dec)
D=6.*u.m
CHORD_channel_width_MHz=0.1953125*u.MHz
def_observing_dec=pi/60.
def_offset=1.75*pi/180. # for this placeholder state where I build up the CHORD layout using rotation matrices instead of actual measurements. probably add Hans' mask at some point to punch the corners and receiver hut holes out...
def_pbw_pert_frac=1e-2
def_evol_restriction_threshold=1./30. # HERA 1/15 was made up. turn this down for a computationally less intense substitute
img_bin_tol=5 # ringing is remarkably insensitive to turning this down; you get really bad scale mismatch by turning it up... the real solution was the "need good resolution in both Fourier and configuration space" thing
def_PA_N_grid_pix=256 # doesn't change the deltaxy; a lower number of pixels per side means eval will be faster
N_fid_beam_types=1
integration_s=10*u.s # seconds
hrs_per_night=8*u.hr # borrowed from Debanjan / 21cmSense
# N_nights=100 # also borrowed from Debanjan / 21cmSense
N_nights=1
# def_N_timesteps=int(N_nights*hrs_per_night//integration_s)
def_N_timesteps=1 # for local tests
print("def_N_timesteps=",def_N_timesteps)

class per_antenna(beam_effects): # still fairly tailored to rectangular arrays
    def __init__(self,
                 mode:str="full",                                                  # run a simulation for full or pathfinder CHORD?
                 b_NS:float=b_NS,b_EW:float=b_EW,                                  # N-S and E-W baseline lengths (m)
                 offset_rad:float=def_offset,                                      # (astropy-unitless because this class expects rad) CHORD is aligned with magnetic, not geographical north, so, when mathematically constructing the uv coverage, rotate the rectangular array grid
                 observing_dec:float=def_observing_dec,                            # declination to observe at (º)
                 N_fiducial_beam_types:int=N_fid_beam_types,N_pert_types:int=0,    # number of fiducial beam types; number of perturbed beam types
                 N_pbws_pert:int=0,                                                # number of antennas with perturbed primary beams
                 pbw_pert_frac:float=def_pbw_pert_frac,                            # ** fractional perturbation to the primary beam width
                 N_timesteps:float=def_N_timesteps,                                # number of timesteps in rotation synthesis
                 nu_ctr:float=nu_HI_z0,                                            # central frequency of the survey of interest
                 pbw_fidu:float=None,                                              # ** fiducial primary beam width (defaults to a diffraction-limited Airy beam, modulo any differences imposed by the number of fiducial beam types)
                 N_grid_pix:int=def_PA_N_grid_pix,                                 # number of pixels per side of the gridded uv plane
                 Delta_nu:float=CHORD_channel_width_MHz,                           # channel width in frequency (MHz)
                 distribution:str="random",                                        # distribution of per-antenna systematics. the options I've encoded for now are random, column, and corner, based on where the fiducial beam types are placed within the array
                 per_channel_systematic=None,                                      # apply a systematic that corrupts the 1/lambda scaling of the beam width? options encoded so far are sporadic (multiply the beam widths for a contiguous chunk of frequency channels by a different multiplicative prefactor for the different fiducial beam types) and D3A-like (noise + too wide at low frequencies... inspired by early three-dish transit beam measurements)
                 evol_restriction_threshold:float=def_evol_restriction_threshold,  # max \delta z/z you will tolerate for the survey of interest and still consider the box close enough to coeval
    
                 sub_ensemble_of_CST_beams=None,                                   # array-like with shape (N_CST_types, N_pointing_errors+1, N_CST_xy, N_CST_xy, N_CST_freqs)
                 CST_xy=None,CST_freqs=None                                        # domain of each CST box in the ensemble. this domain is currently assumed to be the same for each box (not very rigorous/robust, but in practice, if you're running a simulation for a given survey frequency, it would be fairly pathological/ unintuitive/ anti–Occam's razor to get these boxes from CST slices at different frequencies. I guess the practical guidance/takeaway here is that my initial implementation will not support getting different boxes from different CST box resolutions)
                 ): 
                                                                                   # ** args unnecessary for per-antenna CST
        # array and observation geometry
        self.N_fiducial_beam_types=N_fiducial_beam_types
        self.N_pert_types=N_pert_types
        self.N_pbws_pert=N_pbws_pert
        self.pbw_pert_frac=pbw_pert_frac
        self.per_channel_systematic=per_channel_systematic
        self.N_timesteps=N_timesteps
        self.N_grid_pix=N_grid_pix
        self.distribution=distribution
        self.evol_restriction_threshold=evol_restriction_threshold
        self.Delta_nu=Delta_nu
        N_NS=N_NS_full
        N_EW=N_EW_full
        self.DRAO_lat=DRAO_lat
        if (mode=="pathfinder"):
            N_NS=N_NS//2
            N_EW=N_EW//2
        N_ant=N_NS*N_EW
        N_bl=N_ant*(N_ant-1)//2
        self.nu_ctr_MHz=nu_ctr.to(u.MHz)
        self.nu_ctr_Hz=nu_ctr.to(u.Hz)
        self.Dc_ctr=comoving_distance(nu_HI_z0/nu_ctr-1)
        self.N_hrs=hrs_per_night
        self.lambda_obs=c/self.nu_ctr_Hz
        if (pbw_fidu is None):
            pbw_fidu=self.lambda_obs/D
            pbw_fidu=[pbw_fidu,pbw_fidu]
        self.pbw_fidu=np.array(pbw_fidu)
        
        # antenna positions xyz
        antennas_EN=np.zeros((N_ant,2))
        for i in range(N_NS):
            for j in range(N_EW):
                antennas_EN[i*N_EW+j,:]=[j*b_EW.value,i*b_NS.value]
        antennas_EN-=np.mean(antennas_EN,axis=0) # centre the Easting-Northing axes in the middle of the array
        try:
            offset_rad.to(u.rad)
        except:
            offset_rad=offset_rad*u.rad
        offset_from_latlon_rotmat=np.array([[np.cos(offset_rad),-np.sin(offset_rad)],
                                            [np.sin(offset_rad), np.cos(offset_rad)]]) # use this rotation matrix to adjust the NS/EW-only coords
        for i in range(N_ant):
            antennas_EN[i,:]=np.dot(antennas_EN[i,:].T,offset_from_latlon_rotmat)
        dif=antennas_EN[0,0]-antennas_EN[0,-1]+antennas_EN[0,-1]-antennas_EN[-1,-1]
        up=np.reshape(2+(-antennas_EN[:,0]+antennas_EN[:,1])/dif, (N_ant,1), order="C") # eyeballed ~2 m vertical range that ramps ~linearly from a high near the NW corner to a low near the SE corner
        antennas_ENU=np.hstack((antennas_EN,up))
        
        zenith=np.array([np.cos(DRAO_lat),0,np.sin(DRAO_lat)]) # Jon math
        east=np.array([0,1,0])
        north=np.cross(zenith,east)
        lat_mat=np.vstack([north,east,zenith])
        antennas_xyz=antennas_ENU@lat_mat.T
        
        # line-of-sight quantities
        bw_MHz=self.nu_ctr_MHz*evol_restriction_threshold
        N_chan=int(bw_MHz/self.Delta_nu)
        self.N_chan=N_chan
        nu_lo=self.nu_ctr_MHz-bw_MHz/2.
        nu_hi=self.nu_ctr_MHz+bw_MHz/2.
        surv_channels_MHz=np.linspace(nu_hi,nu_lo,N_chan) # decr.
        surv_channels_Hz=surv_channels_MHz.to(u.Hz)
        surv_wavelengths=c/surv_channels_Hz # incr.
        self.surv_wavelengths=surv_wavelengths.decompose()
        z_channels=nu_HI_z0/surv_channels_MHz-1.
        comoving_distances_channels=np.asarray([comoving_distance(chan).value for chan in z_channels]) # incr.
        self.comoving_distances_channels=comoving_distances_channels*u.Mpc
        self.ctr_chan_comov_dist=self.comoving_distances_channels[N_chan//2]
        self.surv_channels_MHz=surv_channels_MHz

        # helper args specific to Gaussian or CST calculations
        CST_Delta_xy=CST_xy[1]-CST_xy[0]
        CST_dxdy=(CST_Delta_xy)**2
        self.CST_dxdy=CST_dxdy
        self.uvbins_CST=fftshift(fftfreq(len(CST_xy),d=CST_Delta_xy))
        self.CST_freqs=CST_freqs
        self.N_CST_xy=len(CST_xy)
        self.N_CST_freqs=len(CST_freqs)

        if type(sub_ensemble_of_CST_beams) is not list: # can't use .ndim because it doesn't behave well for the inhomog arrays of the else
            print("per_antenna received only a !fiducial! beam box")
            fidu_box=sub_ensemble_of_CST_beams
            self.all_boxes=np.expand_dims(sub_ensemble_of_CST_beams,axis=0)
            N_total_beam_types=1
            self.N_total_beam_types=1
        else:
            print("per_antenna received both !fiducial and systematic-laden! beam boxes")
            fidu_box,syst_boxes=sub_ensemble_of_CST_beams # should be unpackable into two arrays:
            assert fidu_box.ndim==3 and syst_boxes.ndim==5 # one box and one "2D array of 3D boxes"
            self.N_CST_types,self.N_max_pointing_errors,Nxy,_,Nz=syst_boxes.shape

            # figure out the actual number of beam types and store the beam types as a list of boxes, not 2D array of boxes + standalone box
            N_pointing_errors_per_CST_case=np.zeros(self.N_CST_types,dtype=int)
            nnn=0
            all_boxes=[fidu_box]
            for i in range(self.N_CST_types):
                for j in range(self.N_max_pointing_errors):
                    box_to_add=syst_boxes[i,j,:,:,:]
                    if not np.all(np.isclose(box_to_add,0.)): # NEW
                        N_pointing_errors_per_CST_case[i]=j # only the (j-1)st case is meaningful, but that is in zero-based indexing, not one-based counting.
                        all_boxes.append(box_to_add)
                        nnn+=1
            self.N_pointing_errors_per_CST_case=N_pointing_errors_per_CST_case
            N_total_beam_types=nnn
            self.N_total_beam_types=N_total_beam_types

            all_boxes=np.asarray(all_boxes)
            self.all_boxes=all_boxes
        self.pb_types=beam_type_distribution(N_NS,N_EW,N_total_beam_types, distribution=self.distribution)

        # ungridded instantaneous uv-coverage (baselines in xyz)
        # second use of the loop: iterate over baselines to make arrays of beam type indices     
        uvw_inst=np.zeros((N_bl,3))
        indices_of_constituent_ant_pb_types=np.zeros((N_bl,2))
        
        k=0
        for i in range(N_ant):
            for j in range(i+1,N_ant):
                uvw_inst[k,:]=antennas_xyz[i,:]-antennas_xyz[j,:]
                indices_of_constituent_ant_pb_types[k]=[self.pb_types[i],self.pb_types[j]]
        
                k+=1
        uvw_inst=np.vstack((uvw_inst,-uvw_inst))
        self.uvw_inst=uvw_inst
        indices_of_constituent_ant_pb_types=np.vstack((indices_of_constituent_ant_pb_types,indices_of_constituent_ant_pb_types)) # get the opposite-permutation baselines for free
        self.indices_of_constituent_ant_pb_types=indices_of_constituent_ant_pb_types
        
        print("computed ungridded instantaneous uv-coverage")

        # rotation-synthesized uv-coverage *******(N_bl,3,N_timesteps), accumulating xyz->uvw transformations at each timestep
        hour_angle_ceiling=np.pi*self.N_hrs/12
        hour_angles=np.linspace(0,hour_angle_ceiling,self.N_timesteps)
        thetas=hour_angles.value*15*np.pi/180*u.rad # don't use built-in astropy conversions for this because it won't realize my hr<->rad conversion is about the rotation rate of the earth
        
        try:
            observing_dec.to(u.rad)
        except:
            observing_dec=observing_dec*u.rad
        zenith=np.array([np.cos(observing_dec),0,np.sin(observing_dec)]) # Jon math redux
        east=np.array([0,1,0])
        north=np.cross(zenith,east)
        project_to_dec=np.vstack([east,north])

        uv_synth=np.zeros((2*N_bl,2,self.N_timesteps))
        for i,theta in enumerate(thetas): # thetas are the rotation synthesis angles (converted from hr. angles using 15 deg/hr rotation rate)
            accumulate_rotation=np.array([[ np.cos(theta),np.sin(theta),0],
                                        [-np.sin(theta),np.cos(theta),0],
                                        [ 0,            0,            1]])
            uvw_rotated=uvw_inst@accumulate_rotation
            uvw_projected=uvw_rotated@project_to_dec.T
            uv_synth[:,:,i]=uvw_projected/self.lambda_obs
        self.uv_synth=uv_synth
        print("synthesized rotation")        

    def calc_uv_coverage(self, Npix:int=1024, pbw_fidu_use:float=None,tol:float=img_bin_tol):
        if pbw_fidu_use is None: # otherwise, use the one that was passed
            pbw_fidu_use=self.pbw_fidu
        all_ungridded_u=self.uv_synth[:,0,:]
        all_ungridded_v=self.uv_synth[:,1,:]
        uvmagmax=tol*np.max([np.max(np.abs(all_ungridded_u)),
                             np.max(np.abs(all_ungridded_v))])

        uvmagmin=2*uvmagmax/Npix
        thetamax=1/uvmagmin # these are 1/-convention Fourier duals, not 2pi/-convention Fourier duals
        self.thetamax=thetamax
        xy_use=self.ctr_chan_comov_dist*np.linspace(-thetamax//2,thetamax//2,Npix)

        uvbins=np.linspace(-uvmagmax,uvmagmax,Npix)
        d2u=uvbins[1]-uvbins[0]
        self.d2u=d2u
        uubins,vvbins=np.meshgrid(uvbins,uvbins, indexing="ij")
        uvplane=np.zeros((Npix,Npix),dtype="complex128") # 0.*uubins
        uvbins_use=np.append(uvbins,uvbins[-1]+uvbins[1]-uvbins[0])
        pad_lo,pad_hi=get_padding(Npix)

        for i in range(self.N_total_beam_types):
            type_i=self.pb_types[i]
            for j in range(i+1):
                type_j=self.pb_types[j]

                here=(self.indices_of_constituent_ant_pb_types[:,0]==i
                        )&(self.indices_of_constituent_ant_pb_types[:,1]==j)
                u_here=self.uv_synth[here,0,:] # [N_bl,2,N_hr_angles]
                v_here=self.uv_synth[here,1,:]
                N_bl_here,N_hr_angles_here=u_here.shape # (N_bl,N_hr_angles)
                N_here=N_bl_here*N_hr_angles_here
                reshaped_u=np.reshape(u_here,N_here,order="C")
                reshaped_v=np.reshape(v_here,N_here,order="C")
                gridded,_,_=np.histogram2d(reshaped_u,reshaped_v,bins=uvbins_use)
                LoS_idx=np.argmin(np.abs(self.nu_obs-self.CST_freqs))
                image_i=self.all_boxes[type_i,:,:,LoS_idx] # [N_total_beam_types, Nxy, Nxy, Nz]
                image_j=self.all_boxes[type_j,:,:,LoS_idx]
                assert np.all(image_i>=0.), "image i beam slice should be entirely nonnegative"
                assert np.all(image_j>=0.), "image j beam slice should be entirely nonnegative"
                image_ij=np.sqrt(image_i*image_j) # geo mean of the beams of this baseline's two constituent antennas. still on initial CST grid
                
                interpolator=RBS(self.CST_xy,self.CST_xy, image_ij)
                image_ij_interpolated=interpolator(xy_use,xy_use)
                plt.figure()
                plt.imshow(image_ij_interpolated.T,origin="lower",norm="log")
                plt.savefig("image_ij.png")
                plt.close()
                kernel=np.abs(fftshift(fftn(ifftshift(image_ij_interpolated*self.CST_dxdy),norm="forward"))) # FT to put in uv space
                
                kernel_padded=np.pad(kernel,((pad_lo,pad_hi),(pad_lo,pad_hi)),"edge")
                convolution_here=convolve(kernel_padded,gridded,mode="valid") # beam-smeared version of the uv-plane for this perturbation permutation
                uvplane+=convolution_here

        uv_bin_edges=[uvbins,uvbins]
        plt.figure()
        plt.imshow(uvplane.T,origin="lower",norm="log")
        plt.savefig("uvplane.png")
        plt.close()
        return uvplane,uv_bin_edges,thetamax # this is the gridded uvplane

    def stack_to_box(self, tol:float=img_bin_tol):
        if (self.nu_ctr_MHz.value<(350/(1-self.evol_restriction_threshold/2)) or 
            self.nu_ctr_MHz>(nu_HI_z0/(1+self.evol_restriction_threshold/2))):
            raise ValueError("{:6.2f} is out of bounds".format(self.nu_ctr_MHz))
        N_grid_pix=self.N_grid_pix
        taper_1d=Blackman_Harris_safe_for_FFT(N_grid_pix) # centre-origin
        taper_x,taper_y=np.meshgrid(taper_1d,taper_1d, indexing="ij")
        self.taper_grid=np.sqrt(taper_x**2+taper_y**2)

        box_uvz=np.zeros((N_grid_pix,N_grid_pix,self.N_chan),dtype="complex128")

        for i in range(self.N_chan): # rescale the uv-coverage to this channel's frequency
            self.uv_synth=self.uv_synth*self.lambda_obs/self.surv_wavelengths[i] # rescale according to observing frequency: multiply up by the prev lambda to cancel, then divide by the current/new lambda
            self.lambda_obs=self.surv_wavelengths[i] # update the observing frequency for next time
            nu_obs=c/self.lambda_obs
            self.nu_obs=nu_obs.decompose()

            # compute the dirty image
            chan_gridded_uvplane,chan_uv_bin_edges,thetamax=self.calc_uv_coverage(Npix=N_grid_pix, tol=tol)
            uv_bin_edges=chan_uv_bin_edges[0]

            # interpolate to store in stack
            if i==0:
                uv_bin_edges_0=chan_uv_bin_edges[0]
                theta_max_box=thetamax
                interpolated_slice=chan_gridded_uvplane
                d2u=self.d2u
            else: # chunk excision and mode interpolation in one step
                interpolator_Re=RBS(uv_bin_edges,uv_bin_edges, chan_gridded_uvplane.real)
                interpolated_slice_Re=interpolator_Re(uv_bin_edges_0,uv_bin_edges_0)
                interpolator_Im=RBS(uv_bin_edges,uv_bin_edges, chan_gridded_uvplane.imag)
                interpolated_slice_Im=interpolator_Im(uv_bin_edges_0,uv_bin_edges_0)
                interpolated_slice=interpolated_slice_Re+1j*interpolated_slice_Im
            # box_uvz[:,:,i]=interpolated_slice
            box_uvz[:,:,i]=interpolated_slice*self.taper_grid
            if ((i%(self.N_chan//3))==0):
                print("{:7.1f} pct complete".format(i/self.N_chan*100))

        box_xyz=fftshift(irfftn(ifftshift(box_uvz*d2u, axes=(0,1)),
                               axes=(0,1),s=(N_grid_pix,N_grid_pix),
                               norm="forward"), axes=(0,1)) # mixed coords before; all config space after
        box_xyz=box_xyz.real # real by construction (mathematically) and coding (irfftn), so stop carrying around the trivial imag part
        for i in range(self.N_chan): # the correct generalization is per-channel normalization
            slice_i=box_xyz[:,:,i]
            norm_i=np.max(slice_i)
            if norm_i>0:
                box_xyz[:,:,i]=slice_i/norm_i # peak-normalize in configuration space
        # box_xyz[box_xyz<0.]=np.abs(box_xyz[box_xyz<0.]) # I tried this on a whim and it was as effective (yet still invalid by construction) a duct tape fix as I could have identified
        self.box=box_xyz

        # generate a box of r-values (necessary for interpolation to survey modes in the manual beam mode of cosmo_stats as called by beam_effects)
        thetas=np.linspace(-theta_max_box,theta_max_box,N_grid_pix)
        xy_vec=self.ctr_chan_comov_dist*thetas # making the coeval approximation
        z_vec=self.comoving_distances_channels-self.ctr_chan_comov_dist 
        self.xy_vec=xy_vec
        self.z_vec=z_vec