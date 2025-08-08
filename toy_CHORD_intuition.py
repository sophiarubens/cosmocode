import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from numpy.fft import ifft2,fftshift,ifftshift
import time

# CHORD layout figures
b_NS=8.5 # m
N_NS=24
b_EW=6.3 # m
N_EW=22
N_ant=N_NS*N_EW
N_bl=N_ant*(N_ant-1)//2
DRAO_lat=49.320791*np.pi/180. # Google Maps satellite view, eyeballing what looks like the middle of the CHORD site: 49.320791, -119.621842 (bc considering drift-scan CHIME-like "pointing at zenith" mode, same as dec)

# general physics stuff
nu_HI_z0=1420.405751768 # MHz
c=2.998e8

# which survey frequencies do I want to examine?
lo=350.        # expected min obs freq
hi=nu_HI_z0    # can't do 21 cm forecasting in the extreme upper end of the CHORD band b/c that would correspond to "blueshifted cosmological HI"
mid=(lo+hi)/2. # midpoint to help connect the dots
obs_freqs=[lo,mid,hi] # MHz

# want to use the part of the Blues colour map that is neither basically white nor basically black
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=1000): # https://stackoverflow.com/questions/18926031/how-to-extract-a-subset-of-a-colormap-as-a-new-colormap-in-matplotlib
    new_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap
cmap = plt.get_cmap("Blues")
trunc_Blues = truncate_colormap(cmap, 0.2, 0.8)

def CHORD_antenna_positions(N_NS=N_NS,N_EW=N_EW,offset_deg=1.75,num_antennas_to_perturb=0,antenna_perturbation_sigma=1e-3):
    N_ant=N_NS*N_EW
    antennas_EN=np.zeros((N_ant,2))
    for i in range(N_NS):
        for j in range(N_EW):
            antennas_EN[i*N_EW+j,:]=[i*N_NS,j*N_EW]
    antennas_EN-=np.mean(antennas_EN,axis=0) # centre the Easting-Northing axes in the middle of the array
    offset=offset_deg*np.pi/180. # actual CHORD is not perfectly aligned to the NS/EW grid. Eyeballed angular offset. -->
    offset_from_latlon_rotmat=np.array([[np.cos(offset),-np.sin(offset)],[np.sin(offset),np.cos(offset)]]) # --> so use this rotation matrix to adjust the NS/EW-only coords
    for i in range(N_ant):
        antennas_EN[i,:]=np.dot(antennas_EN[i,:].T,offset_from_latlon_rotmat)
    dif=antennas_EN[0,0]-antennas_EN[0,-1]+antennas_EN[0,-1]-antennas_EN[-1,-1]
    up=np.reshape(2+(-antennas_EN[:,0]+antennas_EN[:,1])/dif, (N_ant,1)) # eyeballed ~2 m vertical range that ramps ~linearly from a high ~NW corner to a low ~SE corner
    antennas_ENU=np.hstack((antennas_EN,up))
    if (num_antennas_to_perturb>0):
        indices=np.random.randint(0,N_ant,size=num_antennas_to_perturb) # indices of antennas to perturb
        x_perturbations=np.zeros((N_ant,))
        x_perturbations[indices]=np.random.normal(loc=0.,scale=antenna_perturbation_sigma/np.sqrt(3),size=np.insert(num_antennas_to_perturb,0,1))
        y_perturbations=np.zeros((N_ant,))
        y_perturbations[indices]=np.random.normal(loc=0.,scale=antenna_perturbation_sigma/np.sqrt(3),size=np.insert(num_antennas_to_perturb,0,1))
        z_perturbations=np.zeros((N_ant,))
        z_perturbations[indices]=np.random.normal(loc=0.,scale=antenna_perturbation_sigma/np.sqrt(3),size=np.insert(num_antennas_to_perturb,0,1))
        antennas_ENU[:,0]+=x_perturbations
        antennas_ENU[:,1]+=y_perturbations
        antennas_ENU[:,2]+=z_perturbations
    else:
        indices=None
    return antennas_ENU,indices

def uvw_mat(h0,d0=DRAO_lat):
    return np.array([ [ np.sin(h0),             np.cos(h0),            0         ],
                      [-np.sin(d0)*np.cos(h0),  np.sin(d0)*np.sin(h0), np.cos(d0)],
                      [ np.cos(d0)*np.cos(h0), -np.cos(d0)*np.sin(h0), np.sin(d0)]  ])

def calc_baselines_xyz(antennas_ENU,N_NS=N_NS,N_EW=N_EW): # as far as you can go when calculating uv-coverage without needing to know abt frequency
    N_ant=N_NS*N_EW
    N_bl=N_ant*(N_ant-1)//2 # previously: there was a bug where I was calling the number of antennas the number of baselines 
    
    # calculate baselines in ENU
    baselines_ENU=np.zeros((N_bl,3))
    for i in range(N_NS):
        antenna_i=antennas_ENU[i,:]
        for j in range(N_EW):
            k=i*N_EW+j
            antenna_j=antennas_ENU[j,:]
            baseline_ij=antenna_i-antenna_j
            baselines_ENU[  k,   :]=  baseline_ij
            baselines_ENU[-(k+1),:]= -baseline_ij # also accumulate a copy for the version where the other antenna is privileged

    # convert ENU->xyz
    ENU_to_xyz=np.array([[0,-np.sin(DRAO_lat),np.cos(DRAO_lat)],[1,0,0],[0,np.cos(DRAO_lat),np.sin(DRAO_lat)]])
    baselines_xyz=np.zeros((N_bl,3))
    for i in range(N_bl):
        baselines_xyz[i,:]=np.dot(ENU_to_xyz,baselines_ENU[i,:])
    return baselines_xyz

def calc_rot_synth_uv(baselines_xyz,lambda_obs=nu_HI_z0,num_hrs=24,num_timesteps=48): # take [:,:,0] for the instantaneous uv-coverage
    # apply rotation synthesis (accumulate xyz->uvw transformations at each time step)
    hour_angle_ceiling=np.pi*num_hrs/12 # 2pi*num_hrs/24
    hour_angles=np.linspace(0,hour_angle_ceiling,num_timesteps)
    N_bl=baselines_xyz.shape[0]
    uvw_synth=np.zeros((N_bl,3,num_timesteps))
    for i,h0 in enumerate(hour_angles):
        uvw_mat_current=uvw_mat(h0)
        for j in range(N_bl):
            uvw_synth[j,:,i]=np.dot(uvw_mat_current,baselines_xyz[j,:])/lambda_obs
    return uvw_synth

def calc_dirty_image(uvw_synth,nbins=1024): # I remember from my difmap days that you tend to want to anecdotally optimize nbins to be high enough that you get decent resolution but low enough that the Fourier transforms don't take forever, but it would be nice to formalize my logic to get past the point of most of my simulation choices feeling super arbitrary
    N_bl,_,N_hr_angles=uvw_synth.shape
    N_pts_to_bin=N_bl*N_hr_angles
    binned_uvw_synth,_,_=np.histogram2d(np.reshape(uvw_synth[:,0,:],N_pts_to_bin),np.reshape(uvw_synth[:,1,:],N_pts_to_bin),bins=nbins) # [all antennas, x or y, all baselines] ## discarded args are u_edges,v_edges. probably need to shuffle these along at some point in order to properly scale the dirty image axes and not just rely on pixel counts
    return np.abs(fftshift(ifft2(binned_uvw_synth)))
t0=time.time()

N_ant_to_pert=100
ant_pert=1e-2
antennas_ENU_fidu,_=                CHORD_antenna_positions()
antennas_ENU_pert,indices_perturbed=CHORD_antenna_positions(num_antennas_to_perturb=N_ant_to_pert, antenna_perturbation_sigma=ant_pert) # see how many antennas I can get away with perturbing before the difference in the synthesized beam becomes 
t1=time.time()
print("initialized antennas in",t1-t0,"s")

baselines_xyz_fidu=calc_baselines_xyz(antennas_ENU_fidu,N_NS=N_NS,N_EW=N_EW)
baselines_xyz_pert=calc_baselines_xyz(antennas_ENU_pert,N_NS=N_NS,N_EW=N_EW)
t2=time.time()
print("calculated baselines in",t2-t1,"s")

N_obs_hrs=12
uvw_synth_fidu=calc_rot_synth_uv(baselines_xyz_fidu,num_hrs=N_obs_hrs) # precalculate outside the loop and rescale for other frequencies later
uvw_synth_pert=calc_rot_synth_uv(baselines_xyz_pert,num_hrs=N_obs_hrs)
t3=time.time()
print("performed rotation synthesis in",t3-t2,"s")

N_hr_angles=48
colours_b=plt.cm.Blues( np.linspace(1,0.2,N_hr_angles))
lambda_z0=c/(nu_HI_z0*1e6)
tol=1e-16 # near the double precision noise floor
for nu_obs in obs_freqs:
    lambda_obs=c/(nu_obs*1e6)
    z_obs=nu_HI_z0/nu_obs-1.

    # rescale the rotation-synthesized uv coverages to the survey frequency
    uvw_synth_fidu_here=uvw_synth_fidu*lambda_z0/lambda_obs
    uvw_synth_pert_here=uvw_synth_pert*lambda_z0/lambda_obs
    uvw_inst_fidu_here=uvw_synth_fidu_here[:,:,0]
    uvw_inst_pert_here=uvw_synth_pert_here[:,:,0]

    # ift to get dirty images
    dirty_image_fidu=calc_dirty_image(uvw_synth_fidu_here)
    dirty_image_pert=calc_dirty_image(uvw_synth_pert_here)

    # plot
    dotsize=1
    fig,axs=plt.subplots(4,4,figsize=(15,15))
      # general stuff
    for i in range(4):
        axs[i,0].set_xlabel("E (m)")
        axs[i,0].set_ylabel("N (m)")

        axs[i,1].set_xlabel("u ($\lambda$)")
        axs[i,1].set_ylabel("v ($\lambda$)")

        axs[i,2].set_xlabel("u ($\lambda$)")
        axs[i,2].set_ylabel("v ($\lambda$)")

        axs[i,3].set_xlabel("x pixel index")
        axs[i,3].set_ylabel("y pixel index")

      # FIDUCIAL ARRAY
    axs[0,0].scatter(antennas_ENU_fidu[:,0],antennas_ENU_fidu[:,1],s=dotsize,c=antennas_ENU_fidu[:,2],cmap=trunc_Blues)
    axs[0,0].set_title("oversimplified array layout\n (no receiver hut holes, eyeballed array rotation \nand elevation, colour ~ relative U-coord)\nFIDUCIAL ARRAY")

    axs[0,1].scatter(uvw_inst_fidu_here[:,0],uvw_inst_fidu_here[:,1],s=dotsize)
    axs[0,1].set_title("instantaneous uv-coverage/\ndirty beam")

    for i in range(N_hr_angles):
        axs[0,2].scatter(uvw_synth_fidu_here[:,0,i],uvw_synth_fidu_here[:,1,i],color=colours_b[i],s=dotsize) # all baselines, x/y coord, ith time step //one colour = one instance of instantaneous uv-coverage
    axs[0,2].set_title(str(N_obs_hrs)+"-hr rotation-synthesized uv-coverage\nsampled every "+str(int(60/(N_hr_angles/N_obs_hrs)))+" min (colour ~ baseline)")

    im=axs[0,3].imshow(dirty_image_fidu,cmap="Blues",vmax=np.percentile(dirty_image_fidu,99.5))
    plt.colorbar(im,ax=axs[0,3])
    axs[0,3].set_title("dirty image\n(rotation-synthesized uv-coverage \nbinned into "+str(1024)+" bins/axis)")

      # PERTURBED ARRAY
    axs[1,0].scatter(antennas_ENU_pert[:,0],                antennas_ENU_pert[:,1],                s=dotsize,c=antennas_ENU_pert[:,2],                 cmap=trunc_Blues)
    axs[1,0].scatter(antennas_ENU_pert[indices_perturbed,0],antennas_ENU_pert[indices_perturbed,1],s=dotsize,c="r")
    axs[1,0].set_title("PERTURBED ARRAY\nperturbation magnitude="+str(ant_pert*1e3)+"mm")
    axs[1,1].scatter(uvw_inst_pert_here[:,0],uvw_inst_pert_here[:,1],s=dotsize)
    for i in range(N_hr_angles):
        axs[1,2].scatter(uvw_synth_pert_here[:,0,i],uvw_synth_pert_here[:,1,i],color=colours_b[i],s=dotsize)
    im=axs[1,3].imshow(dirty_image_pert,cmap="Blues",vmax=np.percentile(dirty_image_pert,99.5))
    plt.colorbar(im,ax=axs[1,3])

      # RATIOS
    axs[2,0].set_title("RATIO: FIDUCIAL/PERTURBED")
    ratio=dirty_image_fidu/dirty_image_pert
    print("np.max(ratio),np.min(ratio),np.mean(ratio),np.std(ratio)=",np.max(ratio),np.min(ratio),np.mean(ratio),np.std(ratio))
    im=axs[2,3].imshow(ratio,cmap="Blues")
    plt.colorbar(im,ax=axs[2,3])
    # axs[2,3].imshow(np.ma.masked_where(np.abs(ratio-1.)>tol,ratio),cmap="Reds")
    
      # RESIDUALS
    axs[3,0].set_title("RESIDUAL: FIDUCIAL-PERTURBED")
    residual=dirty_image_fidu-dirty_image_pert
    print("np.max(residual),np.min(residual),np.mean(residual),np.std(residual)=",np.max(residual),np.min(residual),np.mean(residual),np.std(residual))
    im=axs[3,3].imshow(residual,cmap="Blues")
    plt.colorbar(im,ax=axs[3,3])
    axs[3,3].imshow(np.ma.masked_where(np.abs(residual)>tol,residual),cmap="Reds")

    plt.suptitle("simulated CHORD-512 observing at "+str(int(nu_obs))+" MHz (z="+str(round(z_obs,3))+")")
    plt.tight_layout()
    plt.savefig("simulated_CHORD_512_"+str(int(nu_obs))+"_MHz_"+str(int(ant_pert*1e3))+"_mm_"+str(int(N_ant_to_pert))+"_ant.png",dpi=200)
    plt.show()