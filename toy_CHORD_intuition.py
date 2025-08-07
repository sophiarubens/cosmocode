import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from numpy.fft import ifft2,fftshift,ifftshift

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=1000):
    new_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

cmap = plt.get_cmap("Blues")
trunc_Blues = truncate_colormap(cmap, 0.2, 0.8)

# CHORD layout inits
b_NS=8.5 # m
N_NS=24
b_EW=6.3 # m
N_EW=22
bminCHORD=6.3
# bmaxCHORD=np.sqrt((b_NS*N_NS)**2+(b_EW*N_EW)**2) # optimistic (includes low-redundancy baselines for CHORD-512) // pf will be 7x10
N_ant=N_NS*N_EW
N_bl=N_ant*(N_ant-1)//2
DRAO_lat=49.320791*np.pi/180. # Google Maps satellite view, eyeballing what looks like the middle of the CHORD site: 49.320791, -119.621842 (bc considering drift-scan CHIME-like "pointing at zenith" mode, same as dec)
nu_HI_z0=1420.405751768 # MHz

lo=350.
hi=nu_HI_z0
mid=(lo+hi)/2.
obs_freqs=[lo,mid,hi] # MHz

for nu_obs in obs_freqs:
    # nu_obs=900. # MHz ... for some reason, the same number I keep using
    c=2.998e8
    lambda_obs=c/(nu_obs*1e6)
    z_obs=nu_HI_z0/nu_obs-1.

    # build array layout
    antennas_EN=np.zeros((N_ant,2))
    for i in range(N_NS):
        for j in range(N_EW):
            antennas_EN[i*N_EW+j,:]=[i*N_NS,j*N_EW]
    antennas_EN-=np.mean(antennas_EN,axis=0) # centre the Easting-Northing axes in the middle of the array
    offset=1.75*np.pi/180. # actual CHORD is not perfectly aligned to the NS/EW grid. Eyeballed angular offset. -->
    offset_from_latlon_rotmat=np.array([[np.cos(offset),-np.sin(offset)],[np.sin(offset),np.cos(offset)]]) # --> so use this rotation matrix to adjust the NS/EW-only coords
    for i in range(N_ant):
        antennas_EN[i,:]=np.dot(antennas_EN[i,:].T,offset_from_latlon_rotmat)
    dif=antennas_EN[0,0]-antennas_EN[0,-1]+antennas_EN[0,-1]-antennas_EN[-1,-1] # x+y00 - x+y-1-1
    up=np.zeros((N_ant,1))                                               # 0th order
    up=np.reshape(2+(-antennas_EN[:,0]+antennas_EN[:,1])/dif, (N_ant,1)) # 1st order 
    antennas_ENU=np.hstack((antennas_EN,up))

    # calc baselines (ENU)
    baselines_ENU=np.zeros((N_bl,3))
    for i in range(N_NS):
        antenna_i=antennas_ENU[i,:]
        for j in range(N_EW):
            k=i*N_EW+j
            antenna_j=antennas_ENU[j,:]
            baseline_ij=antenna_i-antenna_j
            baselines_ENU[  k,   :]=  baseline_ij
            baselines_ENU[-(k+1),:]= -baseline_ij

    # convert ENU->xyz
    ENU_to_xyz=np.array([[0,-np.sin(DRAO_lat),np.cos(DRAO_lat)],[1,0,0],[0,np.cos(DRAO_lat),np.sin(DRAO_lat)]])
    baselines_xyz=np.zeros((N_bl,3))
    for i in range(N_bl):
        baselines_xyz[i,:]=np.dot(ENU_to_xyz,baselines_ENU[i,:])

    # calc inst uv-coverage (convert xyz->uvw)
    def uvw_mat(h0,d0=DRAO_lat):
        return np.array([ [ np.sin(h0),             np.cos(h0),            0         ],
                        [-np.sin(d0)*np.cos(h0),  np.sin(d0)*np.sin(h0), np.cos(d0)],
                        [ np.cos(d0)*np.cos(h0), -np.cos(d0)*np.sin(h0), np.sin(d0)]  ])

    uvw_inst=np.zeros((N_bl,3))
    for i in range(N_bl):
        uvw_inst[i,:]=np.dot(uvw_mat(0),baselines_xyz[i,:])/lambda_obs

    # calc rot synth uv-coverage
    N_h0=48 
    hour_angles=np.linspace(0,np.pi,N_h0)
    colours_b=plt.cm.Blues(np.linspace(1,0.2,N_h0))
    colours_g=plt.cm.Greens(np.linspace(1,0.2,N_bl))
    uvw_synth=np.zeros((N_bl,3,N_h0))
    for i,h0 in enumerate(hour_angles):
        uvw_mat_current=uvw_mat(h0)
        for j in range(N_bl):
            uvw_synth[j,:,i]=np.dot(uvw_mat_current,baselines_xyz[j,:])/lambda_obs

    # ift to get dirty image
    nbins=1024
    binned_uvw_synth,u_edges,v_edges=np.histogram2d(np.reshape(uvw_synth[:,0,:],N_bl*N_h0),np.reshape(uvw_synth[:,1,:],N_bl*N_h0),bins=nbins)
    dirty_image=np.abs(fftshift(ifft2(binned_uvw_synth)))

    dotsize=1
    fig,axs=plt.subplots(1,4,figsize=(15,5))
    axs[0].scatter(antennas_ENU[:,0],antennas_ENU[:,1],s=dotsize,c=up,cmap=trunc_Blues)
    axs[0].set_xlabel("E (m)")
    axs[0].set_ylabel("N (m)")
    axs[0].set_title("oversimplified array layout\n (no holes, eyeballed array rotation and\n elevation, colour ~ relative U-coord)")

    axs[1].scatter(uvw_inst[:,0],uvw_inst[:,1],s=dotsize)
    axs[1].set_xlabel("u ($\lambda$)")
    axs[1].set_ylabel("v ($\lambda$)")
    axs[1].set_title("instantaneous uv-coverage/\ndirty beam")

    for i in range(N_h0):
        axs[2].scatter(uvw_synth[:,0,i],uvw_synth[:,1,i],color=colours_b[i],s=dotsize) # all baselines, x/y coord, ith time step //one colour = one instance of instantaneous uv-coverage
    axs[2].set_xlabel("u ($\lambda$)")
    axs[2].set_ylabel("v ($\lambda$)")
    axs[2].set_title("12-hr rotation-synthesized uv-coverage\nsampled every "+str(int(60/(N_h0/12)))+" min (colour ~ baseline)")

    axs[3].imshow(dirty_image,cmap="Blues",vmax=np.percentile(dirty_image,99.5))
    axs[3].set_xlabel("x pixel index")
    axs[3].set_ylabel("y pixel index")
    axs[3].set_title("dirty image\n(rotation-synthesized uv-coverage \nbinned into "+str(nbins)+" bins/axis)")

    plt.suptitle("simulated CHORD-512 observing at "+str(int(nu_obs))+" MHz (z="+str(round(z_obs,3))+")")
    plt.tight_layout()
    plt.savefig("simulated_CHORD_512_"+str(int(nu_obs))+"_MHz.png",dpi=200)
    plt.show()