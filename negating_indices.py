import numpy as np
from scipy.fft import ifftshift,fftshift

ctr=np.arange(-3,6)
coords=ifftshift(ctr)
print("fftshift(ctr)=",fftshift(ctr))
negcoords=-coords
print("coords=",coords)
# huh=np.nonzero(coords==negcoords)
map=np.argsort(negcoords)[np.argsort(coords)]
print("map=",map)

# negate the coordinates
# find the indices in the original coordinate array that describe the negated coordinates