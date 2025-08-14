import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from numpy.fft import ifft2,fftshift,ifftshift
import time

# CHORD layout params
b_NS=8.5 # m
N_NS=24
b_EW=6.3 # m
N_EW=22
N_ant=N_NS*N_EW
N_bl=N_ant*(N_ant-1)//2
DRAO_lat=49.320791*np.pi/180. # Google Maps satellite view, eyeballing what looks like the middle of the CHORD site: 49.320791, -119.621842 (bc considering drift-scan CHIME-like "pointing at zenith" mode, same as dec)
observing_dec=30.*np.pi/180.

# general physics
nu_HI_z0=1420.405751768 # MHz
c=2.998e8

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

def CHORD_antenna_positions(N_NS=N_NS,N_EW=N_EW,offset_deg=1.75,                    # basics required to specify a CHORD-like array
                            num_antpos_to_perturb=0,antpos_perturbation_sigma=1e-3, # controls for examining the effect of misplaced antennas
                            num_pbs_to_perturb=0,pb_perturbation_sigma=1e-3,        # controls for examining the effect of primary beam width mischaracterizations on a per-antenna basis
                            observatory_latitude=DRAO_lat):       
    N_ant=N_NS*N_EW
    antennas_EN=np.zeros((N_ant,2))
    for i in range(N_NS):
        for j in range(N_EW):
            antennas_EN[i*N_EW+j,:]=[i*N_NS,j*N_EW]
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
    
    zenith=np.array([np.cos(observatory_latitude),0,np.sin(observatory_latitude)])
    east=np.array([0,1,0])
    north=np.cross(zenith,east)
    lat_mat=np.vstack([north,east,zenith])
    antennas_xyz=antennas_ENU@lat_mat.T
    
    ant_pb_frac_widths=np.ones(N_ant)
    if (num_pbs_to_perturb>0):
        indices_pbs=np.random.randint(0,N_ant,size=num_pbs_to_perturb) # indices of antenna pbs to perturb (independent of the indices of antenna positions to perturb, by design)
        pb_perturbations=np.zeros((N_ant,))
        pb_perturbations[indices_pbs]=np.random.normal(loc=0.,scale=pb_perturbation_sigma,size=np.insert(num_pbs_to_perturb,0,1))
        ant_pb_frac_widths+=pb_perturbations
    else:
        indices_pbs=None
    return antennas_xyz,ant_pb_frac_widths,[indices_ants,indices_pbs]

def variable_width_gaussian_primary_beam(u,v,w,ctr,sigma,frac):
    u0,v0,w0=ctr
    sigmau,sigmav,sigmaw=sigma*frac
    evaled=np.exp(-((u-u0)**2/(sigmau**2)+(v-v0)**2/(sigmav**2)+(w-w0)**2/(2*sigmaw**2)))
    return evaled

# def uvw_mat(h0,d0=DRAO_lat):
#     return np.array([ [ np.sin(h0),             np.cos(h0),            0         ],
#                       [-np.sin(d0)*np.cos(h0),  np.sin(d0)*np.sin(h0), np.cos(d0)],
#                       [ np.cos(d0)*np.cos(h0), -np.cos(d0)*np.sin(h0), np.sin(d0)]  ])

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
            pbw[k]=  antenna_pbs[i]*antenna_pbs[j]/np.sqrt(antenna_pbs[i]**2+antenna_pbs[j]**2)
            k+=1
    uvw=np.vstack((uvw,-uvw))
    pbw=np.hstack((pbw,pbw))
    return uvw,pbw

def calc_rot_synth_uv(uvw,lambda_obs=nu_HI_z0,num_hrs=1./2.,num_timesteps=15,dec=30.): # take [:,:,0] for the instantaneous uv-coverage
    """
    * output shape is (N_bl,3,num_timesteps)
    * accumulate xyz->uvw transformations at each time step
    * in dire need of speedup
    """
    hour_angle_ceiling=np.pi*num_hrs/12 # 2pi*num_hrs/24
    hour_angles=np.linspace(0,hour_angle_ceiling,num_timesteps)
    thetas=hour_angles*15*np.pi/180
    
    zenith=np.array([np.cos(dec),0,np.sin(dec)])
    east=np.array([0,1,0])
    north=np.cross(zenith,east)
    project_to_dec=np.vstack([east,north])

    N_bl=uvw.shape[0]
    uv_synth=np.zeros((N_bl,2,num_timesteps))
    for i,theta in enumerate(thetas):
        accumulate_rotation=np.array([[ np.cos(theta),np.sin(theta),0],
                                      [-np.sin(theta),np.cos(theta),0],
                                      [ 0,            0,            1]])
        uvw_rotated=uvw@accumulate_rotation
        uv_synth[:,:,i]=uvw_rotated@project_to_dec.T/lambda_obs
        # uvw_synth[:,:,i]=baselines_xyz@uvw_mat_current/lambda_obs
    return uv_synth

def calc_dirty_image(uvw_synth,baseline_pbs,nbins=1024): # I remember from my difmap days that you tend to want to anecdotally optimize nbins to be high enough that you get decent resolution but low enough that the Fourier transforms don't take forever, but it would be nice to formalize my logic to get past the point of most of my simulation choices feeling super arbitrary
    N_bl,_,N_hr_angles=uvw_synth.shape
    N_pts_to_bin=N_bl*N_hr_angles
    reshaped_u=np.reshape(uvw_synth[:,0,:],N_pts_to_bin)
    reshaped_v=np.reshape(uvw_synth[:,1,:],N_pts_to_bin)
    baseline_pbs_accumulated_reshaped=np.tile(baseline_pbs,N_hr_angles)
    binned_uv_synth,u_edges,v_edges=np.histogram2d(reshaped_u,reshaped_v,bins=nbins,weights=baseline_pbs_accumulated_reshaped) # [all baselines, x or y, all hour angles] ## discarded args are u_edges,v_edges. probably need to shuffle these along at some point in order to properly scale the dirty image axes and not just rely on pixel counts
    # to apply the thing that is kind of like a convolution but actually not (because different antennas get different kernels in the perturbed case), operate on the binned_uv_synth array
    
    dirty_image=np.abs(fftshift(ifft2(binned_uv_synth)))
    return dirty_image,binned_uv_synth,[u_edges,v_edges]

def primary_beam_crossing_time(nu,dec=30.,D=6.):
    beam_width_deg=1.22*(2.998e8/nu)/D*180/np.pi
    crossing_time_hrs_no_dec=beam_width_deg/15
    crossing_time_hrs= crossing_time_hrs_no_dec*np.cos(dec)
    return crossing_time_hrs

"""
SHORTHAND:
fidu = neither antenna nor primary beam perturbations (fiducial array)
antp = yes antenna perturbations, but no primary beam perturbations
prbp = no antenna perturbations, but yes primary beam perturbations
both = both antenna and primary beam perturbations
"""

# currently: laying the groundwork to turn pbs into an actual per_element primary beam perturbation (the p.b. terminology is a bit of an exaggeration for now, but it's just there as a road map/skeleton for what I'm working towards on the shortest timescales)

t0=time.time()
N_ant_to_pert=100
ant_pert=1e-2 
N_pbs_to_pert=100
prb_pert=1e-2
antennas_xyz_fidu,antenna_pbs_fidu,_=                  CHORD_antenna_positions()
np.savetxt("CHORD_ant_pos_unperturbed.txt",antennas_xyz_fidu)
antennas_xyz_antp,antenna_pbs_antp,[indices_antp_a,_]= CHORD_antenna_positions(num_antpos_to_perturb=N_ant_to_pert, antpos_perturbation_sigma=ant_pert)
antennas_xyz_prbp,antenna_pbs_prbp,[_,indices_prbp_p]= CHORD_antenna_positions(num_pbs_to_perturb=N_pbs_to_pert,pb_perturbation_sigma=prb_pert)
antennas_xyz_both,antenna_pbs_both,[indices_both_a,
                                        indices_both_p]=   CHORD_antenna_positions(num_antpos_to_perturb=N_ant_to_pert, antpos_perturbation_sigma=ant_pert,
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
uvw_synth_fidu=calc_rot_synth_uv(baselines_xyz_fidu,num_hrs=N_obs_hrs,num_timesteps=N_hr_angles) # precalculate outside the loop and rescale for other frequencies later
uvw_synth_antp=calc_rot_synth_uv(baselines_xyz_antp,num_hrs=N_obs_hrs,num_timesteps=N_hr_angles)
uvw_synth_prbp=calc_rot_synth_uv(baselines_xyz_prbp,num_hrs=N_obs_hrs,num_timesteps=N_hr_angles)
uvw_synth_both=calc_rot_synth_uv(baselines_xyz_both,num_hrs=N_obs_hrs,num_timesteps=N_hr_angles)
t3=time.time()
print("performed rotation synthesis in",t3-t2,"s")

colours_b=plt.cm.Blues( np.linspace(1,0.2,N_hr_angles))
lambda_z0=c/(nu_HI_z0*1e6)
tol=1e-4 # near the double precision noise floor
for nu_obs in obs_freqs:
    lambda_obs=c/(nu_obs*1e6)
    z_obs=nu_HI_z0/nu_obs-1.

    # rescale the rotation-synthesized uv coverages to the survey frequency
    uvw_synth_fidu_here=uvw_synth_fidu*lambda_z0/lambda_obs
    uvw_synth_antp_here=uvw_synth_antp*lambda_z0/lambda_obs
    uvw_synth_prbp_here=uvw_synth_prbp*lambda_z0/lambda_obs
    uvw_synth_both_here=uvw_synth_both*lambda_z0/lambda_obs
    uvw_inst_fidu_here=uvw_synth_fidu_here[:,:,0]
    uvw_inst_antp_here=uvw_synth_antp_here[:,:,0]
    uvw_inst_prbp_here=uvw_synth_prbp_here[:,:,0]
    uvw_inst_both_here=uvw_synth_both_here[:,:,0]

    # ift to get dirty images
    dirty_image_fidu,binned_uv_synth_fidu,[u_edges_fidu,v_edges_fidu]=calc_dirty_image(uvw_synth_fidu_here,baseline_pbs_fidu)
    dirty_image_antp,binned_uv_synth_antp,[u_edges_antp,v_edges_antp]=calc_dirty_image(uvw_synth_antp_here,baseline_pbs_antp)
    dirty_image_prbp,binned_uv_synth_prbp,[u_edges_prbp,v_edges_prbp]=calc_dirty_image(uvw_synth_prbp_here,baseline_pbs_prbp)
    dirty_image_both,binned_uv_synth_both,[u_edges_both,v_edges_both]=calc_dirty_image(uvw_synth_both_here,baseline_pbs_both)

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
            axs[i,j].set_xlabel("$θ_x$ pixel index")
            axs[i,j].set_ylabel("$θ_y$ pixel index")

      # FIDUCIAL ARRAY
    axs[0,0].scatter(antennas_xyz_fidu[:,0],antennas_xyz_fidu[:,1],s=dotsize,c=antennas_xyz_fidu[:,2],cmap=trunc_Blues)
    axs[0,0].set_title("oversimplified array layout\n (no receiver hut holes,\n eyeballed array rotation and elevation,\n colour ~ relative U-coord)\nFIDUCIAL ARRAY")

    axs[0,1].scatter(uvw_inst_fidu_here[:,0],uvw_inst_fidu_here[:,1],s=dotsize)
    axs[0,1].set_title("instantaneous uv-coverage/\ndirty beam")

    for i in range(N_hr_angles):
        axs[0,2].scatter(uvw_synth_fidu_here[:,0,i],uvw_synth_fidu_here[:,1,i],color=colours_b[i],s=dotsize) # all baselines, x/y coord, ith time step //one colour = one instance of instantaneous uv-coverage
    axs[0,2].set_title(str(round(N_obs_hrs,3))+"-hr rotation-synthesized uv-coverage\nsampled every "+str(int(60/(N_hr_angles/N_obs_hrs)))+" min (colour ~ baseline)")
    im=axs[0,3].imshow(binned_uv_synth_fidu,cmap="Blues",vmin=0.,vmax=np.percentile(binned_uv_synth_fidu,99),origin="lower",
                       extent=[u_edges_fidu[0],u_edges_fidu[-1],v_edges_fidu[0],v_edges_fidu[-1]])
    clb=plt.colorbar(im,ax=axs[0,3])
    clb.ax.set_title("#bl")

    axs[0,3].set_title("binned rotation-synthesized\n and primary-beamed uv-coverage")
    im=axs[0,4].imshow(dirty_image_fidu,cmap="Blues",vmax=np.percentile(dirty_image_fidu,99.5),origin="lower")
    plt.colorbar(im,ax=axs[0,4])
    axs[0,4].set_title("dirty image\n(rotation-synthesized uv-coverage \nbinned into "+str(1024)+" bins/axis)")

      # PERTURBED ANTENNA POSITIONS
    axs[1,0].scatter(antennas_xyz_antp[:,0],           antennas_xyz_antp[:,1],           s=dotsize,c=antennas_xyz_antp[:,2],cmap=trunc_Blues)
    axs[1,0].scatter(antennas_xyz_antp[indices_antp_a,0],antennas_xyz_antp[indices_antp_a,1],s=dotsize,c="r")
    axs[1,0].set_title("PERTURBED ANTENNA POSITIONS\nperturbation magnitude="+str(ant_pert*1e3)+"mm")
    axs[1,1].scatter(uvw_inst_antp_here[:,0],uvw_inst_antp_here[:,1],s=dotsize)
    for i in range(N_hr_angles):
        axs[1,2].scatter(uvw_synth_antp_here[:,0,i],uvw_synth_antp_here[:,1,i],color=colours_b[i],s=dotsize)
    im=axs[1,3].imshow(binned_uv_synth_antp,cmap="Blues",vmin=0.,vmax=np.percentile(binned_uv_synth_antp,99),origin="lower",
                       extent=[u_edges_antp[0],u_edges_antp[-1],v_edges_antp[0],v_edges_antp[-1]])
    clb=plt.colorbar(im,ax=axs[1,3])
    clb.ax.set_title("#bl")
    im=axs[1,4].imshow(dirty_image_antp,cmap="Blues",vmax=np.percentile(dirty_image_antp,99.5),origin="lower")
    plt.colorbar(im,ax=axs[1,4])
    axs[1,5].set_title("ratio: \nfiducial/perturbed")
    ratio=dirty_image_fidu/dirty_image_antp
    im=axs[1,5].imshow(ratio,cmap="Blues",origin="lower",vmax=np.nanpercentile(ratio,99))
    plt.colorbar(im,ax=axs[1,5])
    axs[1,6].set_title("residual: \nfiducial-perturbed")
    residual=dirty_image_fidu-dirty_image_antp
    im=axs[1,6].imshow(residual,cmap="Blues",origin="lower")
    plt.colorbar(im,ax=axs[1,6])
      
      # PERTURBED PRIMARY BEAMS
    axs[2,0].scatter(antennas_xyz_prbp[:,0],             antennas_xyz_prbp[:,1],             s=dotsize,c=antennas_xyz_prbp[:,2],cmap=trunc_Blues)
    axs[2,0].scatter(antennas_xyz_prbp[indices_prbp_p,0],antennas_xyz_prbp[indices_prbp_p,1],s=dotsize,c="tab:orange")
    axs[2,0].set_title("PERTURBED PRIMARY BEAM WIDTHS\nfractional perturbation magnitude="+str(prb_pert))
    axs[2,1].scatter(uvw_inst_antp_here[:,0],uvw_inst_antp_here[:,1],s=dotsize)
    for i in range(N_hr_angles):
        axs[2,2].scatter(uvw_synth_prbp_here[:,0,i],uvw_synth_prbp_here[:,1,i],color=colours_b[i],s=dotsize)
    im=axs[2,3].imshow(binned_uv_synth_prbp,cmap="Blues",vmin=0.,vmax=np.percentile(binned_uv_synth_prbp,99),origin="lower",
                       extent=[u_edges_prbp[0],u_edges_prbp[-1],v_edges_prbp[0],v_edges_prbp[-1]])
    clb=plt.colorbar(im,ax=axs[2,3])
    clb.ax.set_title("#bl")
    im=axs[2,4].imshow(dirty_image_prbp,cmap="Blues",vmax=np.percentile(dirty_image_prbp,99.5),origin="lower")
    plt.colorbar(im,ax=axs[2,4])
    ratio=dirty_image_fidu/dirty_image_prbp
    im=axs[2,5].imshow(ratio,cmap="Blues",origin="lower",vmin=np.nanpercentile(ratio,1),vmax=np.nanpercentile(ratio,99))
    plt.colorbar(im,ax=axs[2,5])
    residual=dirty_image_fidu-dirty_image_prbp
    im=axs[2,6].imshow(residual,cmap="Blues",origin="lower")
    plt.colorbar(im,ax=axs[2,6])

      # PERTURBED ANTENNA POSITIONS *AND* PRIMARY BEAMS
    axs[3,0].scatter(antennas_xyz_both[:,0],           antennas_xyz_both[:,1],           s=dotsize,c=antennas_xyz_both[:,2],cmap=trunc_Blues)
    axs[3,0].scatter(antennas_xyz_both[indices_both_a,0],antennas_xyz_both[indices_both_a,1],s=dotsize,c="r")
    axs[3,0].scatter(antennas_xyz_both[indices_both_p,0],antennas_xyz_both[indices_both_p,1],s=dotsize,c="tab:orange")
    axs[3,0].set_title("PERTURBED ANTENNA POSITIONS\n AND PRIMARY BEAM WIDTHS\ncombined effects of the above cases")
    axs[3,1].scatter(uvw_inst_both_here[:,0],uvw_inst_both_here[:,1],s=dotsize)
    for i in range(N_hr_angles):
        axs[3,2].scatter(uvw_synth_both_here[:,0,i],uvw_synth_both_here[:,1,i],color=colours_b[i],s=dotsize)
    im=axs[3,3].imshow(binned_uv_synth_both,cmap="Blues",vmin=0.,vmax=np.percentile(binned_uv_synth_both,99),origin="lower",
                       extent=[u_edges_both[0],u_edges_both[-1],v_edges_both[0],v_edges_both[-1]])
    clb=plt.colorbar(im,ax=axs[3,3])
    clb.ax.set_title("#bl")
    im=axs[3,4].imshow(dirty_image_both,cmap="Blues",vmax=np.percentile(dirty_image_both,99.5),origin="lower")
    plt.colorbar(im,ax=axs[3,4])
    ratio=dirty_image_fidu/dirty_image_both
    im=axs[3,5].imshow(ratio,cmap="Blues",origin="lower",vmax=np.nanpercentile(ratio,99))
    plt.colorbar(im,ax=axs[3,5])
    residual=dirty_image_fidu-dirty_image_both
    im=axs[3,6].imshow(residual,cmap="Blues",origin="lower")
    plt.colorbar(im,ax=axs[3,6])

    plt.suptitle("simulated CHORD-512 observing at "+str(int(nu_obs))+" MHz (z="+str(round(z_obs,3))+")")
    plt.tight_layout()
    plt.savefig("simulated_CHORD_512_"+str(int(nu_obs))+"_MHz_"+str(int(ant_pert*1e3))+"_mm_"+str(int(N_ant_to_pert))+"_ant.png",dpi=200)
    plt.show()