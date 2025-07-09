import numpy as np
from matplotlib import pyplot as plt
from power import *

# generate_P(T, mode, Lsurvey, Nk0, Nk1=0, V_custom=False)
redshifts=np.arange(10,4,-1)
print("redshifts=",redshifts)
Nbins=12
plt.figure()
for i,z in enumerate(redshifts):
    strz=str(z)
    box=np.load("a_box_z_"+strz+".npy")
    k,P=generate_P(box, "log", 100., Nbins)
    plt.scatter(k,k**3*P/(2.*np.pi**2),label="z="+strz)
    tosave=np.array([k,P]).T
    np.savetxt("z"+strz+"spec.txt",tosave)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("k (1/Mpc)")
plt.ylabel("$\Delta^2$ (mK^2)")
plt.legend()
plt.title("21cmFast box -> power spectrum tests")
plt.savefig("21cmFast_box_to_power_spec_tests.png")
plt.show()