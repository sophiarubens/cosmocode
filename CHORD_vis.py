import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from numpy.fft import ifft2,fftshift,ifftshift
import time
from numba import prange
from scipy.interpolate import interpn
from scipy.signal import convolve
import scipy.sparse as spsp

# CHORD layout params
b_NS=8.5 # m
N_NS=24
b_EW=6.3 # m
N_EW=22
N_ant=N_NS*N_EW
N_bl=N_ant*(N_ant-1)//2
DRAO_lat=49.320791*np.pi/180. # Google Maps satellite view, eyeballing what looks like the middle of the CHORD site: 49.320791, -119.621842 (bc considering drift-scan CHIME-like "pointing at zenith" mode, same as dec)
observing_dec=30.*np.pi/180.

# constants
nu_HI_z0=1420.405751768 # MHz
c=2.998e8
pi=np.pi
twopi=2.*pi
log2=np.log(2)

# survey freqs to examine
lo=350.        # expected min obs freq
hi=nu_HI_z0    # can't do 21 cm forecasting in the extreme upper end of the CHORD band b/c that would correspond to "blueshifted cosmological HI"
mid=(lo+hi)/2. # midpoint to help connect the dots
obs_freqs=[lo,mid,hi] # MHz

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

def CHORD_antenna_positions(b_NS=b_NS,b_EW=b_EW,N_NS=N_NS,N_EW=N_EW,offset_deg=1.75, # basics required to specify a CHORD-like array
                            num_antpos_to_perturb=0,antpos_perturbation_sigma=1e-3,  # controls for examining the effect of misplaced antennas
                            num_pbs_to_perturb=0,pb_perturbation_sigma=1e-3,         # controls for examining the effect of primary beam width mischaracterizations on a per-antenna basis
                            observatory_latitude=DRAO_lat):       
    N_ant=N_NS*N_EW
    antennas_EN=np.zeros((N_ant,2))
    for i in range(N_NS):
        for j in range(N_EW):
            antennas_EN[i*N_EW+j,:]=[j*b_EW,i*b_NS]
    antennas_EN-=np.mean(antennas_EN,axis=0) # centre the Easting-Northing axes in the middle of the array
    offset=offset_deg*np.pi/180. # actual CHORD is not perfectly aligned to the NS/EW grid. Eyeballed angular offset.
    offset_from_latlon_rotmat=np.array([[np.cos(offset),-np.sin(offset)],[np.sin(offset),np.cos(offset)]]) # use this rotation matrix to adjust the NS/EW-only coords
    for i in range(N_ant):
        antennas_EN[i,:]=np.dot(antennas_EN[i,:].T,offset_from_latlon_rotmat)
    dif=antennas_EN[0,0]-antennas_EN[0,-1]+antennas_EN[0,-1]-antennas_EN[-1,-1]
    up=np.reshape(2+(-antennas_EN[:,0]+antennas_EN[:,1])/dif, (N_ant,1)) # eyeballed ~2 m vertical range that ramps ~linearly from a high near the NW corner to a low near the SE corner
    antennas_ENU=np.hstack((antennas_EN,up))
    if (num_antpos_to_perturb>0):
        indices_ants=np.random.randint(0,N_ant,size=num_antpos_to_perturb) # indices of antennas to perturb
        x_perturbations=np.zeros((N_ant,))
        x_perturbations[indices_ants]=np.random.normal(loc=0.,scale=antpos_perturbation_sigma/np.sqrt(3),size=np.insert(num_antpos_to_perturb,0,1))
        y_perturbations=np.zeros((N_ant,))
        y_perturbations[indices_ants]=np.random.normal(loc=0.,scale=antpos_perturbation_sigma/np.sqrt(3),size=np.insert(num_antpos_to_perturb,0,1))
        z_perturbations=np.zeros((N_ant,))
        z_perturbations[indices_ants]=np.random.normal(loc=0.,scale=antpos_perturbation_sigma/np.sqrt(3),size=np.insert(num_antpos_to_perturb,0,1))
        antennas_ENU[:,0]+=x_perturbations
        antennas_ENU[:,1]+=y_perturbations
        antennas_ENU[:,2]+=z_perturbations
    else:
        indices_ants=None
    
    zenith=np.array([np.cos(observatory_latitude),0,np.sin(observatory_latitude)]) # Jon math
    east=np.array([0,1,0])
    north=np.cross(zenith,east)
    lat_mat=np.vstack([north,east,zenith])
    antennas_xyz=antennas_ENU@lat_mat.T
    
    ant_pb_frac_widths=np.ones(N_ant)
    if (num_pbs_to_perturb>0):
        indices_pbs=np.random.randint(0,N_ant,size=num_pbs_to_perturb) # indices of antenna pbs to perturb (independent of the indices of antenna positions to perturb, by design)
        pb_perturbations=np.zeros((N_ant,))
        pb_perturbations[indices_pbs]=pb_perturbation_sigma*np.random.uniform(size=np.insert(num_pbs_to_perturb,0,1))
        ant_pb_frac_widths+=pb_perturbations
    else:
        indices_pbs=None
    return antennas_xyz,ant_pb_frac_widths,[indices_ants,indices_pbs]

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

def calc_inst_uvw(antennas_xyz,antenna_pbs,N_NS=N_NS,N_EW=N_EW):
    N_ant=N_NS*N_EW
    N_bl=N_ant*(N_ant-1)//2
    
    # calculate baselines in xyz (synonymous with instantaneous uv-coverage)
    uvw=np.zeros((N_bl,3))
    pbw=np.zeros(N_bl)
    k=0
    for i in range(N_ant):
        for j in range(i+1,N_ant):
            uvw[k,:]=antennas_xyz[i,:]-antennas_xyz[j,:]
            pbw[k]= 1/np.sqrt( ( (1/antenna_pbs[i]**2)+(1/antenna_pbs[j]**2) )/2. ) # this is for a simple Gaussian beam where the x- and y- FWHMs are the same. Once I get this version working, it should be straightforward to bump up the dimensions and add separate widths
            k+=1
    uvw=np.vstack((uvw,-uvw))
    pbw=np.hstack((pbw,pbw))
    return uvw,pbw

def calc_rot_synth_uv(uvw,lambda_obs=c/(nu_HI_z0*1e6),num_hrs=1./2.,num_timesteps=15,dec=30.): # take [:,:,0] for the instantaneous uv-coverage
    """
    * output shape is (N_bl,3,num_timesteps)
    * accumulate xyz->uvw transformations at each time step
    """
    hour_angle_ceiling=np.pi*num_hrs/12 # 2pi*num_hrs/24
    hour_angles=np.linspace(0,hour_angle_ceiling,num_timesteps)
    thetas=hour_angles*15*np.pi/180
    
    zenith=np.array([np.cos(dec),0,np.sin(dec)]) # Jon math redux
    east=np.array([0,1,0])
    north=np.cross(zenith,east)
    project_to_dec=np.vstack([east,north])

    N_bl=uvw.shape[0]
    uv_synth=np.zeros((N_bl,2,num_timesteps))
    for i,theta in enumerate(thetas): # thetas are the rotation synthesis angles (converted from hr. angles using 15 deg/hr rotation rate)
        accumulate_rotation=np.array([[ np.cos(theta),np.sin(theta),0],
                                      [-np.sin(theta),np.cos(theta),0],
                                      [ 0,            0,            1]])
        uvw_rotated=uvw@accumulate_rotation
        uv_synth[:,:,i]=uvw_rotated@project_to_dec.T/lambda_obs
    return uv_synth

def calc_dirty_image(uv_synth,pbws,primary_beam_width_fidu,Npix=1024,nsigma=5): # I remember from my difmap days that you tend to want to anecdotally optimize nbins to be high enough that you get decent resolution but low enough that the Fourier transforms don't take forever, but it would be nice to formalize my logic to get past the point of most of my simulation choices feeling super arbitrary
    """
    nbins is only used in the first (loopy) step of the not-a-convolution branch
    * the "all primary beam widths are the same" branch always uses nbins_out bins (convolution)
    * the "some perturbed primary beam widths" branch runs the slow, loopy step with nbins bins and then interpolates to nbins_out to facilitate ratios etc. down the line
    """
    N_bl,_,N_hr_angles=uv_synth.shape
    uvmin=np.min([np.min(uv_synth[:,0,:]),np.min(uv_synth[:,1,:])]) # better to deal with a square image
    uvmax=np.max([np.max(uv_synth[:,0,:]),np.max(uv_synth[:,1,:])])
    print("Npix=",Npix)
    thetamax=1/uvmin # these are 1/-convention Fourier duals, not 2pi/-convention Fourier duals
    uvbins=np.linspace(uvmin,uvmax,Npix) # the kind of thing I tended to call "vec" in forecasting_pipeline.py
    d2u=uvbins[1]-uvbins[0]
    uubins,vvbins=np.meshgrid(uvbins,uvbins,indexing="ij") #,sparse=True)
    baselines_w_perturbed_pbs=np.nonzero(pbws!=1)
    N_pe_bl=np.count_nonzero(pbws!=1)
    N_unpe_bl=N_bl-N_pe_bl
    uv_synth_pert_beam=uv_synth[pbws!=1]
    uv_synth_unpe_beam=uv_synth[pbws==1] # have to do it natively pythonically instead of with the np.nonzero output b/c the nonzero output is really only designed to index an array of the same shape as pbws, which is notably not the same shape as the actual list of uv-coverage points
    pbws_pe=pbws[baselines_w_perturbed_pbs]

    fac=1000
    t0=time.time()
    if (N_pe_bl>0):     # handle baselines with    perturbed primary beams using the "accumulate Gaussians" strategy
        # STILL KIND OF LITERAL, BUT NOW TRYING TO SEE IF I CAN BYPASS THE NEED TO INTERPOLATE WITH SPARSE ARRAYS
        uvplane_pert=0.*uubins
        uindices=np.digitize(uv_synth_pert_beam[:,0,:],uvbins)
        vindices=np.digitize(uv_synth_pert_beam[:,1,:],uvbins)
        pix_width=uvbins[1]-uvbins[0]
        for i in range(N_pe_bl):
            pbw_here=pbws_pe[i]*primary_beam_width_fidu

            # figure out how many pixels it takes to get to the n-sigma level (def overkill but idk trying to err on the side of caution here)
            nsigma_distance=nsigma*pbw_here
            nsigma_npix=int(nsigma_distance/pix_width)
            for j in range(N_hr_angles):
                u0,v0=uv_synth_pert_beam[i,:,j]
                smeared_contribution=gaussian_primary_beam_uv(uubins,vvbins,[u0,v0],pbw_here)
                u0i=uindices[i,j] # middle of three original axes collapsed by the indexing fed to digitize
                v0i=vindices[i,j]
                ulo=u0i-nsigma_npix
                uhi=u0i+nsigma_npix
                vlo=v0i-nsigma_npix
                vhi=v0i+nsigma_npix
                uvplane_pert[ulo:uhi,vlo:vhi]+=smeared_contribution[ulo:uhi,vlo:vhi]
            if ((i%(N_pe_bl//fac))==0):
                t1=time.time()
                print(round(i/N_pe_bl*100,5)," pct complete in",t1-t0,"s")
                t0=time.time()

    else:
        uvplane_pert=0.*uubins

    if (N_pe_bl!=N_bl): # handle baselines without perturbed primary beams using the convolution            strategy
        N_pts_to_bin=N_unpe_bl*N_hr_angles
        uvbins_use=np.append(uvbins,uvbins[-1]+uvbins[1]-uvbins[0])
        reshaped_u=np.reshape(uv_synth_unpe_beam[:,0,:],N_pts_to_bin)
        reshaped_v=np.reshape(uv_synth_unpe_beam[:,1,:],N_pts_to_bin)
        uvplane,_,_=np.histogram2d(reshaped_u,reshaped_v,bins=uvbins_use) # version that uses the same bins as in the other case, to make sure the residual and ratio calcs down the line are at least minimally well-conditioned
        kernel=gaussian_primary_beam_uv(uubins,vvbins,[0.,0.,],primary_beam_width_fidu) # doesn't matter which beam width you take bc they're all the same in this branch... could probably make things more efficient down the line by only carting around the single value in the case where the beam widths are all the same
        pad_lo,pad_hi=get_padding(Npix)
        kernel_padded=np.pad(kernel,((pad_lo,pad_hi),(pad_lo,pad_hi)),"edge")
        uvplane_unpe=convolve(kernel_padded,uvplane,mode="valid") # beam-smeared version of the uv-plane
    else:
        uvplane_unpe=0.*uubins
    
    uvplane=uvplane_pert+uvplane_unpe
    dirty_image=np.abs(fftshift(ifft2(uvplane*d2u,norm="forward")))
    uv_bin_edges=[uvbins,uvbins]
    theta_lims=[-thetamax,thetamax,-thetamax,thetamax]
    return dirty_image,uvplane,uv_bin_edges,theta_lims

def primary_beam_crossing_time(nu,dec=30.,D=6.):
    beam_width_deg=1.22*(2.998e8/nu)/D*180/np.pi
    crossing_time_hrs_no_dec=beam_width_deg/15
    crossing_time_hrs= crossing_time_hrs_no_dec*np.cos(dec)
    return crossing_time_hrs

"""
SHORTHAND:
fidu = neither antenna nor primary beam perturbations (fiducial array)
antp = antenna perturbations, but no primary beam perturbations
prbp = no antenna perturbations, but yes primary beam perturbations
both = both antenna and primary beam perturbations
"""

t0=time.time()
N_ant_to_pert=100
ant_pert=1e-2 
N_pbs_to_pert=100
prb_pert=1e-2
antennas_xyz_fidu,antenna_pbs_fidu,_=                    CHORD_antenna_positions()
np.savetxt("CHORD_ant_pos_unperturbed.txt",antennas_xyz_fidu)

antennas_xyz_antp,antenna_pbs_antp,[indices_antp_a,_]=   CHORD_antenna_positions(num_antpos_to_perturb=N_ant_to_pert, antpos_perturbation_sigma=ant_pert)
antennas_xyz_prbp,antenna_pbs_prbp,[_,indices_prbp_p]=   CHORD_antenna_positions(num_pbs_to_perturb=N_pbs_to_pert,pb_perturbation_sigma=prb_pert)
antennas_xyz_both,antenna_pbs_both,[indices_both_a,
                                        indices_both_p]= CHORD_antenna_positions(num_antpos_to_perturb=N_ant_to_pert, antpos_perturbation_sigma=ant_pert,
                                                                                 num_pbs_to_perturb=N_pbs_to_pert,pb_perturbation_sigma=prb_pert)
t1=time.time()
print("initialized antennas in",t1-t0,"s")

baselines_xyz_fidu,baseline_pbs_fidu=calc_inst_uvw(antennas_xyz_fidu,antenna_pbs_fidu,N_NS=N_NS,N_EW=N_EW)
baselines_xyz_antp,baseline_pbs_antp=calc_inst_uvw(antennas_xyz_antp,antenna_pbs_antp,N_NS=N_NS,N_EW=N_EW)
baselines_xyz_prbp,baseline_pbs_prbp=calc_inst_uvw(antennas_xyz_prbp,antenna_pbs_prbp,N_NS=N_NS,N_EW=N_EW)
baselines_xyz_both,baseline_pbs_both=calc_inst_uvw(antennas_xyz_both,antenna_pbs_both,N_NS=N_NS,N_EW=N_EW)
t2=time.time()
print("calculated baselines in",t2-t1,"s")

N_obs_hrs=primary_beam_crossing_time(350e6,dec=DRAO_lat+30.) # longest possible crossing time is for a source observed at the bottom of the CHORD band
print("N_obs_hrs=",N_obs_hrs) # looking at the worst-case scenario to get something plausible while still avoiding doing rotation synthesis cals inside the loop
N_hr_angles=15
uv_synth_fidu=calc_rot_synth_uv(baselines_xyz_fidu,num_hrs=N_obs_hrs,num_timesteps=N_hr_angles) # precalculate outside the loop and rescale for other frequencies later
uv_synth_antp=calc_rot_synth_uv(baselines_xyz_antp,num_hrs=N_obs_hrs,num_timesteps=N_hr_angles)
uv_synth_prbp=calc_rot_synth_uv(baselines_xyz_prbp,num_hrs=N_obs_hrs,num_timesteps=N_hr_angles)
uv_synth_both=calc_rot_synth_uv(baselines_xyz_both,num_hrs=N_obs_hrs,num_timesteps=N_hr_angles)
t3=time.time()
print("performed rotation synthesis in",t3-t2,"s")

colours_b=plt.cm.Blues( np.linspace(1,0.2,N_hr_angles))
lambda_z0=c/(nu_HI_z0*1e6)
tol=1e-4 # near the double precision noise floor
for nu_obs in obs_freqs:
    lambda_obs=c/(nu_obs*1e6)
    z_obs=nu_HI_z0/nu_obs-1.
    p_b_width_fidu=1.22*lambda_obs/6.*2*np.sqrt(2*log2) # in radians (yeah ok CHORD dish diameter hard-coded here)
    print("p_b_width_fidu=",p_b_width_fidu)

    # rescale the rotation-synthesized uv coverages to the survey frequency
    uv_synth_fidu_here=uv_synth_fidu*lambda_z0/lambda_obs
    uv_synth_antp_here=uv_synth_antp*lambda_z0/lambda_obs
    uv_synth_prbp_here=uv_synth_prbp*lambda_z0/lambda_obs
    uv_synth_both_here=uv_synth_both*lambda_z0/lambda_obs
    uvw_inst_fidu_here=uv_synth_fidu_here[:,:,0]
    uvw_inst_antp_here=uv_synth_antp_here[:,:,0]
    uvw_inst_prbp_here=uv_synth_prbp_here[:,:,0]
    uvw_inst_both_here=uv_synth_both_here[:,:,0]

    # ift to get dirty images
    ta=time.time()
    dirty_image_fidu,binned_uv_synth_fidu,[u_edges_fidu,v_edges_fidu],[thetaxmin_fidu,thetaxmax_fidu,thetaymin_fidu,thetaymax_fidu]=calc_dirty_image(uv_synth_fidu_here,baseline_pbs_fidu,p_b_width_fidu)
    tb=time.time()
    print("dirty_image_fidu evaluated in",tb-ta,"s")
    dirty_image_antp,binned_uv_synth_antp,[u_edges_antp,v_edges_antp],[thetaxmin_antp,thetaxmax_antp,thetaymin_antp,thetaymax_antp]=calc_dirty_image(uv_synth_antp_here,baseline_pbs_antp,p_b_width_fidu)
    tc=time.time()
    print("dirty_image_antp evaluated in",tc-tb,"s")
    dirty_image_prbp,binned_uv_synth_prbp,[u_edges_prbp,v_edges_prbp],[thetaxmin_prbp,thetaxmax_prbp,thetaymin_prbp,thetaymax_prbp]=calc_dirty_image(uv_synth_prbp_here,baseline_pbs_prbp,p_b_width_fidu)
    td=time.time()
    print("dirty_image_prbp evaluated in",td-tc,"s")
    dirty_image_both,binned_uv_synth_both,[u_edges_both,v_edges_both],[thetaxmin_both,thetaxmax_both,thetaymin_both,thetaymax_both]=calc_dirty_image(uv_synth_both_here,baseline_pbs_both,p_b_width_fidu)
    te=time.time()
    print("dirty_image_both evaluated in",te-td,"s")

    # plot
    dotsize=1
    fig,axs=plt.subplots(4,7,figsize=(31,15))
    for i in range(4):
        axs[i,0].set_xlabel("E (m)")
        axs[i,0].set_ylabel("N (m)")

        for j in range(1,4):
            axs[i,j].set_xlabel("u ($\lambda$)")
            axs[i,j].set_ylabel("v ($\lambda$)")

        for j in range(4,7):
            axs[i,j].set_xlabel("$θ_x$ (rad)")
            axs[i,j].set_ylabel("$θ_y$ (rad)")

      # FIDUCIAL ARRAY
    axs[0,0].scatter(antennas_xyz_fidu[:,0],antennas_xyz_fidu[:,1],s=dotsize,c=antennas_xyz_fidu[:,2],cmap=trunc_Blues)
    axs[0,0].set_title("oversimplified array layout\n (no receiver hut holes,\n eyeballed array rotation and elevation,\n colour ~ relative U-coord)\nFIDUCIAL ARRAY")

    axs[0,1].scatter(uvw_inst_fidu_here[:,0],uvw_inst_fidu_here[:,1],s=dotsize)
    axs[0,1].set_title("instantaneous uv-coverage/\ndirty beam")

    for i in range(N_hr_angles):
        axs[0,2].scatter(uv_synth_fidu_here[:,0,i],uv_synth_fidu_here[:,1,i],color=colours_b[i],s=dotsize) # all baselines, x/y coord, ith time step //one colour = one instance of instantaneous uv-coverage
    axs[0,2].set_title(str(round(N_obs_hrs,3))+"-hr rotation-synthesized uv-coverage\nsampled every "+str(int(60/(N_hr_angles/N_obs_hrs)))+" min (colour ~ baseline)")
    im=axs[0,3].imshow(binned_uv_synth_fidu,cmap="Blues",vmin=np.percentile(binned_uv_synth_fidu,1),vmax=np.percentile(binned_uv_synth_fidu,99),origin="lower",
                       extent=[u_edges_fidu[0],u_edges_fidu[-1],v_edges_fidu[0],v_edges_fidu[-1]])
    clb=plt.colorbar(im,ax=axs[0,3])
    clb.ax.set_title("#bl")

    axs[0,3].set_title("binned rotation-synthesized\n and primary-beamed uv-coverage")
    im=axs[0,4].imshow(dirty_image_fidu,cmap="Blues",vmin=np.percentile(dirty_image_fidu,2),vmax=np.percentile(dirty_image_fidu,98),origin="lower",
                       extent=[thetaxmin_fidu,thetaxmax_fidu,thetaymin_fidu,thetaymax_fidu])
    plt.colorbar(im,ax=axs[0,4])
    axs[0,4].set_title("dirty image\n(rotation-synthesized uv-coverage \nbinned into "+str(1024)+" bins/axis)")

      # PERTURBED ANTENNA POSITIONS
    axs[1,0].scatter(antennas_xyz_antp[:,0],           antennas_xyz_antp[:,1],           s=dotsize,c=antennas_xyz_antp[:,2],cmap=trunc_Blues)
    axs[1,0].scatter(antennas_xyz_antp[indices_antp_a,0],antennas_xyz_antp[indices_antp_a,1],s=dotsize,c="r")
    axs[1,0].set_title("PERTURBED ANTENNA POSITIONS\nperturbation magnitude="+str(ant_pert*1e3)+"mm")
    axs[1,1].scatter(uvw_inst_antp_here[:,0],uvw_inst_antp_here[:,1],s=dotsize)
    for i in range(N_hr_angles):
        axs[1,2].scatter(uv_synth_antp_here[:,0,i],uv_synth_antp_here[:,1,i],color=colours_b[i],s=dotsize)
    im=axs[1,3].imshow(binned_uv_synth_antp,cmap="Blues",vmin=np.percentile(binned_uv_synth_antp,1),vmax=np.percentile(binned_uv_synth_antp,99),origin="lower",
                       extent=[u_edges_antp[0],u_edges_antp[-1],v_edges_antp[0],v_edges_antp[-1]])
    clb=plt.colorbar(im,ax=axs[1,3])
    clb.ax.set_title("#bl")
    im=axs[1,4].imshow(dirty_image_antp,cmap="Blues",vmin=np.percentile(dirty_image_antp,2),vmax=np.percentile(dirty_image_antp,98),origin="lower",
                       extent=[thetaxmin_antp,thetaxmax_antp,thetaymin_antp,thetaymax_antp])
    plt.colorbar(im,ax=axs[1,4])
    axs[1,5].set_title("ratio: \nfiducial/perturbed")
    ratio=dirty_image_fidu/dirty_image_antp
    im=axs[1,5].imshow(ratio,cmap="Blues",origin="lower",vmax=np.nanpercentile(ratio,99),
                       extent=[thetaxmin_antp,thetaxmax_antp,thetaymin_antp,thetaymax_antp])
    plt.colorbar(im,ax=axs[1,5])
    axs[1,6].set_title("residual: \nfiducial-perturbed")
    residual=dirty_image_fidu-dirty_image_antp
    im=axs[1,6].imshow(residual,cmap="Blues",origin="lower",
                       extent=[thetaxmin_antp,thetaxmax_antp,thetaymin_antp,thetaymax_antp])
    plt.colorbar(im,ax=axs[1,6])
      
      # PERTURBED PRIMARY BEAMS
    axs[2,0].scatter(antennas_xyz_prbp[:,0],             antennas_xyz_prbp[:,1],             s=dotsize,c=antennas_xyz_prbp[:,2],cmap=trunc_Blues)
    axs[2,0].scatter(antennas_xyz_prbp[indices_prbp_p,0],antennas_xyz_prbp[indices_prbp_p,1],s=dotsize,c="tab:orange")
    axs[2,0].set_title("PERTURBED PRIMARY BEAM WIDTHS\nfractional perturbation magnitude="+str(prb_pert))
    axs[2,1].scatter(uvw_inst_antp_here[:,0],uvw_inst_antp_here[:,1],s=dotsize)
    for i in range(N_hr_angles):
        axs[2,2].scatter(uv_synth_prbp_here[:,0,i],uv_synth_prbp_here[:,1,i],color=colours_b[i],s=dotsize)
    im=axs[2,3].imshow(binned_uv_synth_prbp,cmap="Blues",vmin=np.percentile(binned_uv_synth_prbp,1),vmax=np.percentile(binned_uv_synth_prbp,99),origin="lower",
                       extent=[u_edges_prbp[0],u_edges_prbp[-1],v_edges_prbp[0],v_edges_prbp[-1]])
    clb=plt.colorbar(im,ax=axs[2,3])
    clb.ax.set_title("#bl")
    im=axs[2,4].imshow(dirty_image_prbp,cmap="Blues",vmin=np.percentile(dirty_image_prbp,2),vmax=np.percentile(dirty_image_prbp,98),origin="lower",
                       extent=[thetaxmin_prbp,thetaxmax_prbp,thetaymin_prbp,thetaymax_prbp])
    plt.colorbar(im,ax=axs[2,4])
    ratio=dirty_image_fidu/dirty_image_prbp
    im=axs[2,5].imshow(ratio,cmap="Blues",origin="lower",vmin=np.nanpercentile(ratio,1),vmax=np.nanpercentile(ratio,99),
                       extent=[thetaxmin_prbp,thetaxmax_prbp,thetaymin_prbp,thetaymax_prbp])
    plt.colorbar(im,ax=axs[2,5])
    residual=dirty_image_fidu-dirty_image_prbp
    im=axs[2,6].imshow(residual,cmap="Blues",origin="lower",
                       extent=[thetaxmin_prbp,thetaxmax_prbp,thetaymin_prbp,thetaymax_prbp])
    plt.colorbar(im,ax=axs[2,6])

      # PERTURBED ANTENNA POSITIONS *AND* PRIMARY BEAMS
    axs[3,0].scatter(antennas_xyz_both[:,0],           antennas_xyz_both[:,1],           s=dotsize,c=antennas_xyz_both[:,2],cmap=trunc_Blues)
    axs[3,0].scatter(antennas_xyz_both[indices_both_a,0],antennas_xyz_both[indices_both_a,1],s=dotsize,c="r")
    axs[3,0].scatter(antennas_xyz_both[indices_both_p,0],antennas_xyz_both[indices_both_p,1],s=dotsize,c="tab:orange")
    axs[3,0].set_title("PERTURBED ANTENNA POSITIONS\n AND PRIMARY BEAM WIDTHS\ncombined effects of the above cases")
    axs[3,1].scatter(uvw_inst_both_here[:,0],uvw_inst_both_here[:,1],s=dotsize)
    for i in range(N_hr_angles):
        axs[3,2].scatter(uv_synth_both_here[:,0,i],uv_synth_both_here[:,1,i],color=colours_b[i],s=dotsize)
    im=axs[3,3].imshow(binned_uv_synth_both,cmap="Blues",vmin=np.percentile(binned_uv_synth_both,1),vmax=np.percentile(binned_uv_synth_both,99),origin="lower",
                       extent=[u_edges_both[0],u_edges_both[-1],v_edges_both[0],v_edges_both[-1]])
    clb=plt.colorbar(im,ax=axs[3,3])
    clb.ax.set_title("#bl")
    im=axs[3,4].imshow(dirty_image_both,cmap="Blues",vmin=np.percentile(dirty_image_both,2),vmax=np.percentile(dirty_image_both,98),origin="lower",
                       extent=[thetaxmin_both,thetaxmax_both,thetaymin_both,thetaymax_both])
    plt.colorbar(im,ax=axs[3,4])
    ratio=dirty_image_fidu/dirty_image_both
    im=axs[3,5].imshow(ratio,cmap="Blues",origin="lower",vmax=np.nanpercentile(ratio,99),
                       extent=[thetaxmin_both,thetaxmax_both,thetaymin_both,thetaymax_both])
    plt.colorbar(im,ax=axs[3,5])
    residual=dirty_image_fidu-dirty_image_both
    im=axs[3,6].imshow(residual,cmap="Blues",origin="lower",
                       extent=[thetaxmin_both,thetaxmax_both,thetaymin_both,thetaymax_both])
    plt.colorbar(im,ax=axs[3,6])

    plt.suptitle("simulated CHORD-512 observing at "+str(int(nu_obs))+" MHz (z="+str(round(z_obs,3))+")")
    plt.tight_layout()
    plt.savefig("simulated_CHORD_512_"+str(int(nu_obs))+"_MHz_"+str(int(ant_pert*1e3))+"_mm_"+str(int(N_ant_to_pert))+"_ant.png",dpi=200)
    plt.show()

    # assert(1==0), "refining workflow with one frequency por ahora"