import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from CHORD_vis import *
from cosmo_distances import *
import time

test_freq=363
lambda_obs=c/(test_freq*1e6)
nu_HI_z0=1420.405751768 # MHz
z_obs=nu_HI_z0/test_freq-1.
deltanu=0.183
Npix=256

fidu=CHORD_image(nu_ctr=test_freq, N_pert_types=0)
pert=CHORD_image(nu_ctr=test_freq, N_pert_types=2,  num_pbws_to_pert=100)

t0=time.time()
fidu.stack_to_box(delta_nu=deltanu,N_grid_pix=Npix)
fidu_box=fidu.box
fidu_theta_max=fidu.thetamax
N_grid_pix=fidu.N_grid_pix
np.save("fidu_box_"+str(int(test_freq))+"_"+str(Npix)+".npy",fidu_box)
print("fidu theta_max=",fidu_theta_max)
t1=time.time()
print("built fiducially-beamed box in",t1-t0,"s")
pert.stack_to_box(delta_nu=deltanu,N_grid_pix=Npix)
pert_box=pert.box
pert_theta_max=pert.thetamax # should be the same as the other one bc of survey freq things... although there might be slight numerical differences if you start moving antennas or the region of support for one of the perturbed beams is very skewed
np.save("pert_box_"+str(int(test_freq))+"_"+str(Npix)+".npy", pert_box)
print("pert theta_max=",pert_theta_max)
t2=time.time()
print("built perturbed-beamed box in",t2-t1,"s")