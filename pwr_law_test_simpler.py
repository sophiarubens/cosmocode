import numpy as np
from matplotlib import pyplot as plt

def elbowy_power(k,a=0.96605,b=-0.8,c=1,a0=1,b0=5000):
    return c/(a0*k**(-a)+b0*k**(-b))

# changing b changes the slope of how quickly the power law reaches the high-k quasi-asymptote
# a0 and b0 control the L/R shift of the elbow

npts=150
k=np.linspace(1e-3,1,npts)
P=elbowy_power(k)
plt.figure()
tests=np.linspace(5000,1,5)
for i in tests:
    plt.loglog(k,elbowy_power(k,b0=i),label="b0="+str(i))
plt.legend()
plt.title("intuition for an elbowy power law\nP(k)=c/(a0*k**(-a)+b0*k**(-b))\na=0.99605, b=-0.8, c=a0=1, vary b0")
plt.xlabel("k")
plt.ylabel("P(k)")
plt.tight_layout()
plt.savefig("elbowy_pwr_law_intuition.png")
plt.show()