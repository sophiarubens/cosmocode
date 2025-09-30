import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from numpy.fft import ifft2,fftshift #,ifftshift
import time
from scipy.signal import convolve
import scipy.sparse as spsp
from scipy.interpolate import interpn

# CHORD immutables
N_NS_full=24
N_EW_full=22
DRAO_lat=49.320791*np.pi/180. # Google Maps satellite view, eyeballing what looks like the middle of the CHORD site: 49.320791, -119.621842 (bc considering drift-scan CHIME-like "pointing at zenith" mode, same as dec)
D=6. # m

# constants
pi=np.pi
nu_HI_z0=1420.405751768 # MHz
c=2.998e8
twopi=2.*pi
log2=np.log(2)

class SurveyOutOfBoundsError(Exception):
    pass

class CHORD_image(object):
    def __init__(self,
                 mode="full",b_NS=8.5,b_EW=6.3,observing_dec=pi/60.,offset_deg=1.75*pi/180.,N_pert_types=4,
                 num_ant_pos_to_pert=0,ant_pos_pert_sigma=1e-2,
                 num_pbws_to_pert=0,pbw_pert_sigma=1e-2,
                 num_timesteps=15,num_hrs=None,
                 nu_ctr=nu_HI_z0,
                 pbw_fidu=None
                 ):
        # array and observation geometry
        self.N_NS=N_NS_full
        self.N_EW=N_EW_full
        self.DRAO_lat=DRAO_lat
        if (mode=="pathfinder"):
            self.N_NS=self.N_NS//2
            self.N_EW=self.N_EW//2
        self.N_ant=self.N_NS*self.N_EW
        self.N_bl=self.N_ant*(self.N_ant-1)//2
        self.observing_dec=observing_dec
        self.num_timesteps=num_timesteps
        self.nu_ctr=nu_ctr
        if (num_hrs is None):
            num_hrs=primary_beam_crossing_time(self.nu_ctr*1e6,dec=self.observing_dec,D=D) # frew needs to be in Hz
        self.num_hrs=num_hrs
        self.lambda_obs=c/self.nu_ctr
        if (pbw_fidu is None):
            pbw_fidu=self.lambda_obs/D
        self.pbw_fidu=pbw_fidu
        self.ant_pos_pert_sigma=ant_pos_pert_sigma
        self.pbw_pert_sigma=pbw_pert_sigma
        
        # antenna positions xyz
        antennas_EN=np.zeros((self.N_ant,2))
        for i in range(self.N_NS):
            for j in range(self.N_EW):
                antennas_EN[i*self.N_EW+j,:]=[j*b_EW,i*b_NS]
        antennas_EN-=np.mean(antennas_EN,axis=0) # centre the Easting-Northing axes in the middle of the array
        offset=offset_deg*pi/180. # actual CHORD is not perfectly aligned to the NS/EW grid. Eyeballed angular offset.
        offset_from_latlon_rotmat=np.array([[np.cos(offset),-np.sin(offset)],[np.sin(offset),np.cos(offset)]]) # use this rotation matrix to adjust the NS/EW-only coords
        for i in range(self.N_ant):
            antennas_EN[i,:]=np.dot(antennas_EN[i,:].T,offset_from_latlon_rotmat)
        dif=antennas_EN[0,0]-antennas_EN[0,-1]+antennas_EN[0,-1]-antennas_EN[-1,-1]
        up=np.reshape(2+(-antennas_EN[:,0]+antennas_EN[:,1])/dif, (self.N_ant,1)) # eyeballed ~2 m vertical range that ramps ~linearly from a high near the NW corner to a low near the SE corner
        antennas_ENU=np.hstack((antennas_EN,up))
        if (num_ant_pos_to_pert>0):
            indices_ant_pos_pert=np.random.randint(0,self.N_ant,size=num_ant_pos_to_pert) # indices of antennas to perturb
            x_perturbations=np.zeros((self.N_ant,))
            x_perturbations[indices_ant_pos_pert]=np.random.normal(loc=0.,scale=ant_pos_pert_sigma/np.sqrt(3),size=np.insert(num_ant_pos_to_pert,0,1))
            y_perturbations=np.zeros((self.N_ant,))
            y_perturbations[indices_ant_pos_pert]=np.random.normal(loc=0.,scale=ant_pos_pert_sigma/np.sqrt(3),size=np.insert(num_ant_pos_to_pert,0,1))
            z_perturbations=np.zeros((self.N_ant,))
            z_perturbations[indices_ant_pos_pert]=np.random.normal(loc=0.,scale=ant_pos_pert_sigma/np.sqrt(3),size=np.insert(num_ant_pos_to_pert,0,1))
            antennas_ENU[:,0]+=x_perturbations
            antennas_ENU[:,1]+=y_perturbations
            antennas_ENU[:,2]+=z_perturbations
        else:
            indices_ant_pos_pert=None
        self.indices_ant_pos_pert=indices_ant_pos_pert
        
        zenith=np.array([np.cos(DRAO_lat),0,np.sin(DRAO_lat)]) # Jon math
        east=np.array([0,1,0])
        north=np.cross(zenith,east)
        lat_mat=np.vstack([north,east,zenith])
        antennas_xyz=antennas_ENU@lat_mat.T
        self.antennas_xyz=antennas_xyz

        pbw_types=np.zeros((self.N_ant,))
        self.N_pert_types=N_pert_types
        N_beam_types=N_pert_types+1
        self.N_beam_types=N_beam_types
        epsilons=np.zeros(N_beam_types)
        if (num_pbws_to_pert>0):
            epsilons[1:]=pbw_pert_sigma*np.random.uniform(size=np.insert(N_pert_types,0,1))
            indices_of_ants_w_pert_pbws=np.random.randint(0,self.N_ant,size=num_pbws_to_pert) # indices of antenna pbs to perturb (independent of the indices of antenna positions to perturb, by design)
            pbw_types[indices_of_ants_w_pert_pbws]=np.random.randint(1,high=N_beam_types,size=np.insert(num_pbws_to_pert,0,1)) # leaves as zero the indices associated with unperturbed antennas
        else:
            indices_of_ants_w_pert_pbws=None
        self.pbw_types=pbw_types
        self.indices_of_ants_w_pert_pbws=indices_of_ants_w_pert_pbws
        self.epsilons=epsilons
        
        # ungridded instantaneous uv-coverage (baselines in xyz)        
        uvw_inst=np.zeros((self.N_bl,3))
        indices_of_constituent_ant_pb_types=np.zeros((self.N_bl,2))
        k=0
        for i in range(self.N_ant):
            for j in range(i+1,self.N_ant):
                uvw_inst[k,:]=antennas_xyz[i,:]-antennas_xyz[j,:]
                indices_of_constituent_ant_pb_types[k]=[pbw_types[i],pbw_types[j]] # 1/np.sqrt( ( (1/antenna_pbs[i]**2)+(1/antenna_pbs[j]**2) )/2. ) # this is for a simple Gaussian beam where the x- and y- FWHMs are the same. Once I get this version working, it should be straightforward to bump up the dimensions and add separate widths
                k+=1
        uvw_inst=np.vstack((uvw_inst,-uvw_inst))
        indices_of_constituent_ant_pb_types=np.vstack((indices_of_constituent_ant_pb_types,indices_of_constituent_ant_pb_types))
        self.uvw_inst=uvw_inst
        self.indices_of_constituent_ant_pb_types=indices_of_constituent_ant_pb_types
        print("computed ungridded instantaneous uv-coverage")

        # rotation-synthesized uv-coverage *******(N_bl,3,N_timesteps), accumulating xyz->uvw transformations at each timestep
        hour_angle_ceiling=np.pi*num_hrs/12 # 2pi*num_hrs/24
        hour_angles=np.linspace(0,hour_angle_ceiling,num_timesteps)
        thetas=hour_angles*15*np.pi/180
        
        zenith=np.array([np.cos(self.observing_dec),0,np.sin(self.observing_dec)]) # Jon math redux
        east=np.array([0,1,0])
        north=np.cross(zenith,east)
        project_to_dec=np.vstack([east,north])

        uv_synth=np.zeros((2*self.N_bl,2,num_timesteps))
        for i,theta in enumerate(thetas): # thetas are the rotation synthesis angles (converted from hr. angles using 15 deg/hr rotation rate)
            accumulate_rotation=np.array([[ np.cos(theta),np.sin(theta),0],
                                        [-np.sin(theta),np.cos(theta),0],
                                        [ 0,            0,            1]])
            uvw_rotated=uvw_inst@accumulate_rotation
            uv_synth[:,:,i]=uvw_rotated@project_to_dec.T/self.lambda_obs
        self.uv_synth=uv_synth
        print("synthesized rotation")

    def calc_dirty_image(self, Npix=1024, pbw_fidu_use=None):
        if pbw_fidu_use is None: # otherwise, use the one that was passed
            pbw_fidu_use=self.pbw_fidu
        t0=time.time()
        abs_uv_synth=np.abs(self.uv_synth)
        uvmin=np.min([np.min(abs_uv_synth[:,0,:]),np.min(abs_uv_synth[:,1,:])]) # better to deal with a square image
        self.uvmin=uvmin
        uvmax=np.max([np.max(abs_uv_synth[:,0,:]),np.max(abs_uv_synth[:,1,:])])
        self.uvmax=uvmax
        thetamax=1/uvmin # these are 1/-convention Fourier duals, not 2pi/-convention Fourier duals
        self.thetamax=thetamax
        uvbins=np.linspace(uvmin,uvmax,Npix) # the kind of thing I tended to call "vec" in forecasting_pipeline.py
        d2u=uvbins[1]-uvbins[0]
        uubins,vvbins=np.meshgrid(uvbins,uvbins,indexing="ij")

        uvplane=0.*uubins
        uvbins_use=np.append(uvbins,uvbins[-1]+uvbins[1]-uvbins[0])
        pad_lo,pad_hi=get_padding(Npix)
        for i in range(self.N_beam_types):
            eps_i=self.epsilons[i]
            for j in range(i+1):
                eps_j=self.epsilons[j]
                here=(self.indices_of_constituent_ant_pb_types[:,0]==i)&(self.indices_of_constituent_ant_pb_types[:,1]==j) # which baselines to treat during this loop trip... pbws has shape (N_bl,2) ... one column for antenna a and the other for antenna b
                u_here=self.uv_synth[here,0,:] # [N_bl,3,N_hr_angles]
                v_here=self.uv_synth[here,1,:]
                N_bl_here,N_hr_angles_here=u_here.shape # (N_bl,N_hr_angles)
                N_here=N_bl_here*N_hr_angles_here
                reshaped_u=np.reshape(u_here,N_here)
                reshaped_v=np.reshape(v_here,N_here)
                gridded,_,_=np.histogram2d(reshaped_u,reshaped_v,bins=uvbins_use)
                width_here=pbw_fidu_use*np.sqrt((1-eps_i)*(1-eps_j))
                kernel=gaussian_primary_beam_uv(uubins,vvbins,[0.,0.],width_here)
                kernel_padded=np.pad(kernel,((pad_lo,pad_hi),(pad_lo,pad_hi)),"edge")
                convolution_here=convolve(kernel_padded,gridded,mode="valid") # beam-smeared version of the uv-plane for this perturbation permutation
                uvplane+=convolution_here

        uvplane/=self.N_beam_types**2 # divide out the artifact of there having been multiple convolutions
        self.uvplane=uvplane
        dirty_image=np.abs(fftshift(ifft2(uvplane*d2u,norm="forward")))
        uv_bin_edges=[uvbins,uvbins]
        t1=time.time()
        print("computed dirty image in ",t1-t0,"s")
        self.dirty_image=dirty_image
        self.uv_bin_edges=uv_bin_edges
        self.thetamax=thetamax # eventually get rid of this redundant block once I fully remove the attribute-baked checks of per_antenna_permuts.py
        return dirty_image,uv_bin_edges,thetamax

    def stack_to_box(self,delta_nu,evol_restriction_threshold=1./15., N_grid_pix=1024):
        if (self.nu_ctr<(350/(1-evol_restriction_threshold/2)) or self.nu_ctr>(nu_HI_z0/(1+evol_restriction_threshold/2))):
            raise SurveyOutOfBoundsError
        self.N_grid_pix=N_grid_pix
        bw=self.nu_ctr*evol_restriction_threshold
        N_chan=int(bw/delta_nu)
        self.nu_lo=self.nu_ctr-bw/2.
        self.nu_hi=self.nu_ctr+bw/2.
        # surv_channels=np.linspace(self.nu_hi,self.nu_lo,N_chan) # descending
        surv_channels=np.linspace(self.nu_lo,self.nu_hi,N_chan)
        surv_wavelengths=c/surv_channels # ascending
        surv_beam_widths=surv_wavelengths/D # ascending (need to traverse the beam widths in ascending order in order to use the 0th entry to set the excision cross-section)
        self.surv_channels=surv_channels
        box=np.zeros((N_chan,N_grid_pix,N_grid_pix))
        print("N_chan, surv_channels.shape=",N_chan,surv_channels.shape)
        for i,beam_width in enumerate(surv_beam_widths):
            # rescale the uv-coverage to this channel's frequency
            self.uv_synth=self.uv_synth*self.lambda_obs/surv_wavelengths[i] # rescale according to observing frequency: multiply up by the prev lambda to cancel, then divide by the current/new lambda
            self.lambda_obs=surv_wavelengths[i] # update the observing frequency for next time

            # compute the dirty image
            chan_dirty_image,chan_uv_bin_edges,thetamax=self.calc_dirty_image(Npix=N_grid_pix, pbw_fidu_use=beam_width)
            # print("thetamax=",thetamax)
            
            # interpolate to store in stack
            if i==0:
                uv_bin_edges_0=chan_uv_bin_edges[0]
                uu_bin_edges_0,vv_bin_edges_0=np.meshgrid(uv_bin_edges_0,uv_bin_edges_0,indexing="ij")
                theta_max=thetamax
                interpolated_slice=chan_dirty_image
            else:
                # print("chan_uv_bin_edges=",chan_uv_bin_edges)
                # print("uu_bin_edges_0=",uu_bin_edges_0)
                # chunk excision and interpolation in one step:
                interpolated_slice=interpn(chan_uv_bin_edges,
                                           chan_dirty_image,
                                           (uu_bin_edges_0,vv_bin_edges_0),
                                           bounds_error=False, fill_value=None) # extrap necessary because the smallest u and v you have at a given slice-needing-extrapolation will be larger than the min u and v mags to extrapolate to
            box[i]=interpolated_slice
            if ((i%(N_chan//10))==0):
                print("{:5}%% complete".format(i/N_chan*100))
        self.box=box # it would be lowkey diabolical to send this to cosmo_stats to window numerically and expect to generate box realizations at the same resolution
        self.theta_max=theta_max

# use the part of the Blues colour map with decent contrast and eyeball-ably perceivable differences between adjacent samplings
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=1000): # https://stackoverflow.com/questions/18926031/how-to-extract-a-subset-of-a-colormap-as-a-new-colormap-in-matplotlib
    new_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap
cmap = plt.get_cmap("Blues")
trunc_Blues = truncate_colormap(cmap, 0.2, 0.8)

def get_padding(n):
    padding=n-1
    padding_lo=int(np.ceil(padding / 2))
    padding_hi=padding-padding_lo
    return padding_lo,padding_hi

def gaussian_primary_beam_uv(u,v,ctr,fwhm):
    u0,v0=ctr
    evaled=((pi*log2)/(fwhm**2))*np.exp(-pi**2*(((u-u0)**2+(v-v0)**2)*fwhm**2)/np.log(2))
    # fwhmx,fwhmy=fwhm
    # evaled=((pi*log2)/(fwhmx*fwhmy))*np.exp(-pi**2*((u-u0)**2*fwhmx**2+(v-v0)**2*fwhmy**2)/np.log(2))
    return evaled

def sparse_gaussian_primary_beam_uv(u,v,ctr,fwhm,nsigma_npix):
    """
    same as the non-sparse version but uses scipy sparse arrays to make things less inefficient

    u,v  - square coordinate arrays defining the grid
    ctr  - uv coordinates of beam peak
    fwhm -  
    """
    # figure out where to put the Gaussian and its values
    u0,v0=ctr
    # will need indices of the peak of the beam in the uv plane for sparse array anchoring purposes
    base=0.*u
    evaled=((pi*log2)/(fwhm**2))*np.exp(-pi**2*(((u-u0)**2+(v-v0)**2)*fwhm**2)/np.log(2))
    u0i,v0i=np.unravel_index(evaled.argmax(), evaled.shape)
    base[u0i-nsigma_npix:u0i+nsigma_npix,v0i-nsigma_npix:v0i+nsigma_npix]=evaled[u0i-nsigma_npix:u0i+nsigma_npix,v0i-nsigma_npix:v0i+nsigma_npix]
    evaled_sparse=spsp.csr_array(base)

    # mask the 10-sigma region and store as a sparse array
    return evaled_sparse

def primary_beam_crossing_time(nu,dec=30.,D=6.):
    beam_width_deg=1.22*(2.998e8/nu)/D*180/np.pi
    crossing_time_hrs_no_dec=beam_width_deg/15
    crossing_time_hrs= crossing_time_hrs_no_dec*np.cos(dec)
    return crossing_time_hrs