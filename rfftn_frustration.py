import numpy as np
from scipy.fft import rfftn, irfftn, fftn

rng=np.random.default_rng()
s=(3,5)
a=rng.normal(size=s)
print("a.shape=",a.shape)
A_real=rfftn(a,s=s,axes=(0,1))
A_regu=fftn(a,axes=(0,1))
print("A_regu.shape=",A_regu.shape)
print(A_real==A_regu[:3,:3])