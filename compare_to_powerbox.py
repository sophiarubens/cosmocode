import numpy as np
from matplotlib import pyplot as plt
from power import *
from bias_helper_fcns import *
from power_class import *
import time

import powerbox as pbox
from powerbox.powerbox import _magnitude_grid
from powerbox.dft import fft,ifft # normalization and Fourier dual characterized by (a,b)=(0,2pi) (the cosmology convention), although the numpy convention is (a,b)=(1,1)

# Parameters of the field
# L = 10. # this example seems precisely tuned to give ~machine precision errors (pseudorandom other things I've tried do not yield such results)
# N = 512
L = 126
N = 512
dx = L/N

# POWERBOX BOX GEN
x = np.arange(-L/2,L/2,dx)[:N] # The 1D field grid
r = _magnitude_grid(x,dim=3)   # The magnitude of the co-ordinates on a 3D grid
field = np.exp(-np.pi*r**2)    # Create the field

# Generate the k-space field, the 1D k-space grid, and the 3D magnitude grid.
k_field, k, rk = fft(field,L=L,          # Pass the field to transform, and its size
                     ret_cubegrid=True   # Tell it to return the grid of magnitudes.
                    )
k_arr=np.array(k)

# Plot the field minus the analytic result
inspect=13
plt.figure()
plt.imshow(np.abs(k_field)[:,:,inspect]-np.exp(-np.pi*rk**2)[:,:,inspect],extent=(k_arr.min(),k_arr.max(),k_arr.min(),k_arr.max()))
plt.colorbar()
plt.title("powerbox Gaussian residuals")
plt.savefig("powerbox_fft_residuals.png")
plt.show()

# MY BOX GEN (ANALOGOUS CASE)

# ACCUMULATE REALIZATIONS OF POWERBOX WAY VS MY WAY

# COMPARE MOMENTS OF POWERBOX WAY VS MY WAY