import numpy as np
from matplotlib import pyplot as plt
from forecasting_pipeline import *

save_CST=True
beam_sim_directory="/Users/sophiarubens/Downloads/research/code/pipeline/CST_beams_Aditya/CHORD_Fiducial_200kHz/"
freq_lo=0.58 # GHz for file names
freq_hi=0.62 
delta_nu_CST=0.0002
freqs_for_xis=np.arange(freq_lo,freq_hi,delta_nu_CST)
zs_for_xis=[nu_HI_z0/freq-1 for freq in freqs_for_xis]
xis=[comoving_distance(z) for z in zs_for_xis]

boxname="classified_fiducial"
gen_box=True
if gen_box:
    test=reconfigure_CST_beam(freq_lo,freq_hi,delta_nu_CST,beam_sim_directory=beam_sim_directory,
                              box_outname=boxname,Nxy=64)
    test.gen_box_from_simulated_beams()
    test_box=test.box
    xy_vec=test.xy_for_box
    z_vec=test.CST_z_vec
    np.save(beam_sim_directory+"z_vec.npy",z_vec)
else:
    test_box=np.load(beam_sim_directory+"CST_box_"+boxname+".npy")
    xy_vec=np.load(beam_sim_directory+"xy_vec_for_box"+boxname+".npy") 
    z_vec=np.load(beam_sim_directory+"z_vec.npy")

print("box_test.shape=",test_box.shape)

# examine some slices to see if mínimamente estoy barking up the right tree
fig,axs=plt.subplots(4,3,figsize=(12,12),layout="constrained")
for j in range(4):
    N0,N1,N2=test_box.shape

    cut0=int(N0*j/4)
    yz_slice=test_box[cut0,:,:]
    im=axs[j,0].imshow(yz_slice.T,origin="lower", vmax=1,
                       extent=[xy_vec[0],xy_vec[-1],z_vec[0],z_vec[-1]])
    plt.colorbar(im,ax=axs[j,0])
    axs[j,0].set_xlabel("y (Mpc)")
    axs[j,0].set_ylabel("z (Mpc)")
    axs[j,0].set_title(str(cut0)+"/"+str(N2)+" yz")

    cut1=int(N1*j/4)
    xz_slice=test_box[:,cut1,:]
    im=axs[j,1].imshow(xz_slice.T,origin="lower", vmax=1,
                       extent=[xy_vec[0],xy_vec[-1],z_vec[0],z_vec[-1]])
    plt.colorbar(im,ax=axs[j,1])
    axs[j,1].set_xlabel("x (Mpc)")
    axs[j,1].set_ylabel("z (Mpc)")
    axs[j,1].set_title(str(cut1)+"/"+str(N2)+" xz")

    cut2=int(N2*j/4)
    xy_slice=test_box[:,:,cut2]
    im=axs[j,2].imshow(xy_slice.T,origin="lower", vmax=1,
                       extent=[xy_vec[0],xy_vec[-1],xy_vec[0],xy_vec[-1]])
    plt.colorbar(im,ax=axs[j,2])
    axs[j,2].set_xlabel("x (Mpc)")
    axs[j,2].set_ylabel("y (Mpc)")
    axs[j,2].set_title(str(cut2)+"/"+str(N2)+" xy")
    plt.savefig("CST_classified_beam_slices.png")