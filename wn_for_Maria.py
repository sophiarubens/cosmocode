from forecasting_pipeline import cosmo_stats
import numpy as np
from matplotlib import pyplot as plt
from astropy import units as u
from matplotlib.colors import CenteredNorm

N=100
wn=cosmo_stats(100*u.Mpc,
               P_fid=np.ones(20)*u.mK**2*u.Mpc**3,
               k_fid=np.linspace(1e-3,10,20)/u.Mpc,
               Nvox=N,Nvoxz=N)
wn.generate_GRF()
box=wn.T_pristine
np.save("white_noise_for_Maria"+str(N)+".npy",box.value)

wn.power_Monte_Carlo()
plt.imshow(wn.P_binned.T,origin="lower",norm=CenteredNorm(vcenter=1.))
plt.xlabel("k-perp index")
plt.ylabel("k-par index")
plt.colorbar()
plt.savefig("white_noise_for_Maria_"+str(N)+".png")