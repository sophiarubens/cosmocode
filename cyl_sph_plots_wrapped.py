import numpy as np
from cosmo_distances import *
from forecasting_pipeline import *

redo_window_calc=True
mode="pathfinder" # "pathfinder", "full"
nu_ctr=363. 
frac_tol_conv=0.1
N_sph=256 # how many spherical modes to put in your theory power spectrum or bin final power spectra down to
ceil=90

N_fidu_types=4
N_pert_types=2
N_pbws_pert=40
per_channel_systematic="sporadic" # None, "D3A_like", "sporadic"
PA_dist="corner" # "random", "corner"
epsxy=0.05

plot_qty="P"
f_types_prefacs=np.linspace(0.85,1.15,N_fidu_types) # trivial for now, but it will be less trivial later
categ="PA" # "PA", "UAA"
uaa_beam_type="Airy" # only used in UAA mode... probably make this less hacky, but por ahora es lo que hay

cyl_sph_plots(redo_window_calc,
              mode, nu_ctr, epsxy,
              ceil, frac_tol_conv, N_sph,
              categ, uaa_beam_type, 
              N_fidu_types, N_pert_types, 
              N_pbws_pert, per_channel_systematic,
              PA_dist, f_types_prefacs, plot_qty)