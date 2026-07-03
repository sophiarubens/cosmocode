import numpy as np
from scipy.fft import fft, fftfreq, fftshift,ifftshift
from matplotlib import pyplot as plt
import mpmath
# Claude integral. Mathematica, my knowledge of integration by parts and other special techniques, etc. did not suffice. 

twopi=2*np.pi
mpmath.mp.dps=128
def analytical(k,a,b,alpha=-3.3):
    s=alpha+1
    prefac=mpmath.power(1j*k,-s)
    return prefac*(mpmath.gammainc(s,1j*k*a)-mpmath.gammainc(s,1j*k*b)) # difference of upper incomplete gamma functions set by integration bounds

# freq range in box = 590, 610 MHz
a, b = 590, 610 # upper and lower lims in config space. pipeline case from 12 Jun AM print: 

pipeline_foreground_box=np.load("fg_power.npy")
pipeline_foreground_means=ifftshift(np.mean(pipeline_foreground_box,axis=(0,1)))
N_pl = len(pipeline_foreground_means)
N=2048
# N=N_pl
print("N=",N)
x = np.linspace(a, b, N)
dx = x[1]-x[0]
k_coords=twopi*fftfreq(N, d=dx) # all evaluated at pos config space points = pos freq components = no need to fftshift
k_coords_pl=twopi*fftfreq(N_pl,d=(b-a)/N_pl)

# power_laws= [ [-2.8, 335.4],
            #   [-2.15, 33.5]  ] # unless I also carry over the freq channels this normalization will not be of immediate use in this verification script
power_laws= [ [-2.8, 335.4]  ]
total_F_analytical=np.zeros(N,dtype="complex128")
total_F_numerical=np.zeros(N,dtype="complex128")
for power_law in power_laws:
    alpha,amplitude=power_law
    f=x**alpha
    F_numerical=fft(f)*dx 

    F_analytical=np.zeros(N,dtype="complex128")
    for i,k in enumerate(k_coords): # mpmath handles one value at a time. the price to pay for precision numerics...
        F_analytical[i]=analytical(k, a, b, alpha=alpha)
    total_F_analytical+=F_analytical
    total_F_numerical+=F_numerical

FT2_numerical=np.abs(total_F_numerical)**2
FT2_analytical=np.abs(total_F_analytical)**2
plt.figure()
plt.loglog(np.abs(k_coords),FT2_numerical,
           ls="dotted",label="numerical")
plt.loglog(np.abs(k_coords),FT2_analytical,
           ls="dashed",label="analytical")
plt.loglog(np.abs(k_coords_pl),pipeline_foreground_means/pipeline_foreground_means[0]*FT2_numerical[0],
           marker=".",label="pipeline")
plt.legend()
plt.xlabel("k")
plt.ylabel("unnormalized power")
plt.title("power law power comparison")
plt.savefig("power_comparison.png")