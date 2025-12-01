import numpy as np
from cosmo_distances import *
from forecasting_pipeline import *

redo_window_calc=True
redo_box_calc=True

contaminant_or_window=None # None (calculates contaminant power), "window"
mode="pathfinder" # "pathfinder", "full"
nu_ctr=900. 
frac_tol_conv=0.01
N_sph=256 # how many spherical modes to put in your theory power spectrum or bin final power spectra down to

uaa_beam_type="Airy" # only used in UAA mode... probably make this less hacky, but por ahora es lo que hay
N_fidu_types=1
N_pert_types=1
categ="PA" # "PA", "UAA"
N_pbws_pert=0
per_channel_systematic="sporadic" # None, "D3A_like", "sporadic"
PA_dist="random" # "random", "corner"
f_types_prefacs=[1.] # np.linspace(0.85,1.15,N_fidu_types) # for a trivial case: [1.] 
plot_qty="P"
epsxy=0.1

cyl_sph_plots(redo_window_calc,
              redo_box_calc,
              mode, nu_ctr, epsxy,
              0, frac_tol_conv, N_sph, # ceil=0 always now
              categ, uaa_beam_type, 
              N_fidu_types, N_pert_types, 
              N_pbws_pert, per_channel_systematic,
              PA_dist, f_types_prefacs, plot_qty,
              contaminant_or_window=contaminant_or_window, k_idx_for_window=136,
              isolated=False)