import numpy as np
from scipy.fft import fft,ifft,fftshift,ifftshift
from matplotlib import pyplot as plt

d3r=1 # placeholder for testing
effective_volume=1 # placeholder for testing
fg_column=np.load("fg_column.npy") # centre
N_fg=len(fg_column)
fg_column=np.ones(N_fg)
P_unbinned=np.abs( fftshift( fft( ifftshift(
           fg_column)*d3r ) ) )**2/effective_volume # centre
P_doctored=np.copy(P_unbinned)
P_doctored[N_fg//2]=np.nan

fig,axs=plt.subplots(1,3,layout="constrained")
axs[0].plot(fg_column)
axs[0].set_title("FG LoS column")
axs[1].plot(P_unbinned)
axs[1].set_title("abs(fftshift(fft(\nifftshift(FG LoS \ncolumn))))**2")
axs[2].plot(P_doctored) 
axs[2].set_title("inset of \nunnormalized \npower spec")
# axs[2].set_ylim(-1e2,1e8) # ok for actual fg
# axs[2].set_xlim(98,106)

# axs[2].set_ylim(-1e2,1e3) # ok for ones test
# axs[2].set_xlim(100,104)
for i in range(3):
    axs[i].set_xlabel("LoS voxel index")
    axs[i].set_ylabel("power (arbitrary units)")
plt.savefig("fg_column_inspection.png")