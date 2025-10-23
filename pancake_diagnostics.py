import numpy as np
from matplotlib import pyplot as plt

phead="/Users/sophiarubens/Downloads/research/code/"
ptail="_363_256.npy"
twopi=2*np.pi

xy_vec_box=np.load(phead+"per_antenna/xy_vec_for_boxes"+ptail)
z_vec_box=np.load(phead+"per_antenna/z_vec_for_boxes"+ptail)

kpar_min_box= twopi/(z_vec_box[-1]-  z_vec_box[0])
kpar_max_box= twopi/(z_vec_box[-1]-  z_vec_box[-2])
kpar_vec_box=np.linspace(kpar_min_box,kpar_max_box,len(xy_vec_box))
kperp_min_box=twopi/(xy_vec_box[-1]-xy_vec_box[0])
kperp_max_box=twopi/(xy_vec_box[-1]-xy_vec_box[-2])
kperp_vec_box=np.linspace(kperp_min_box,kperp_max_box,len(z_vec_box))

kx_grid_box,ky_grid_box,kz_grid_box=np.meshgrid(kperp_vec_box,kperp_vec_box,kpar_vec_box)
k_grid_box=np.sqrt(kx_grid_box**2+ky_grid_box**2+kz_grid_box**2)

kpar_surv=np.load(phead+"pipeline/kpar_surv"+ptail)
kperp_surv=np.load(phead+"pipeline/kperp_surv"+ptail)
kmin