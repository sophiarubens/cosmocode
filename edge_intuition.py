import numpy as np
from numpy.fft import fft,fftfreq
from matplotlib import pyplot as plt

N=357
edge=1
x=np.linspace(-3,2,N)
y=np.zeros(N)
y[edge:-edge]=1
# y=np.ones(N)
X=fftfreq(N)
Y=fft(y)
f=(x-0.5)**2
f[:edge]=0
f[-edge:]=0
F=fft(f)

# want to deconvolve in Fourier space so divide in config space
# deconvolve?
F_decontam=fft(f/F)

fig,axs=plt.subplots(3,3,figsize=(6,6),layout="constrained")
# top row for x,y, X,Y
axs[0,0].plot(x,y,c="C0")
axs[0,0].set_title("config space rectangle")
axs[0,1].scatter(X,Y.real,    label="amp",   s=1,c="C1")
axs[0,1].scatter(X,Y.imag,    label="phase", s=1,c="C2")
axs[0,1].scatter(X,np.abs(Y), label="norm",  s=1,c="C3")
axs[0,1].set_ylim(-3,10)
axs[0,1].legend()
axs[0,1].set_title("harmonic space sinc = \nFT(config space rectangle)")

# middle row for x,f, X,F raw 
axs[1,0].plot(x,f,c="C0")
axs[1,0].set_title("function evaluated over\nconfig space rectangle")
axs[1,1].scatter(X,F.real,    label="amp",   s=1,c="C1")
axs[1,1].scatter(X,F.imag,    label="phase", s=1,c="C2")
axs[1,1].scatter(X,np.abs(F), label="norm",  s=1,c="C3")
axs[1,1].set_ylim(-3,50)
axs[1,1].legend()
axs[1,1].set_title("FT(function evaluated over\nconfig space rectangle)")

# bottom row for x,f, X,F decontaminated 
axs[2,2].scatter(X,F_decontam.real,    label="amp",   s=1,c="C4")
axs[2,2].scatter(X,F_decontam.imag,    label="phase", s=1,c="C5")
axs[2,2].scatter(X,np.abs(F_decontam), label="norm",  s=1,c="C6")
axs[2,2].set_ylim(-1,5)
axs[2,2].set_title("DECONTAMINATED function in harmonic space")
plt.title("numerical Fourier intuition")
plt.savefig("edge_intuition.png")