import numpy as np
from matplotlib import pyplot as plt
from scipy.special import j0,j1,jv

pi=np.pi
twopi=2.*pi

def airybeam(theta,alpha=1):
    return ((j1(alpha*theta))/(alpha*theta))**2

npts=2222
theta_vals=np.linspace(0,twopi,npts)
# j0_0=j0(0)
# j1_0=j1(0)
# j2_0=jv(2,0)
# print('j0_0,j1_0,j2_0=',j0_0,j1_0,j2_0)
basic_airy_beam=airybeam(theta_vals)
airy_beam_0=1./4.
print("airy_beam_0=",airy_beam_0)
basic_airy_beam_half_max=airy_beam_0/2.
beta_fwhm=theta_vals[np.nanargmin(np.abs(basic_airy_beam-basic_airy_beam_half_max))]
print("basic_airy_beam_fwhm*180/pi=",beta_fwhm*180/pi)

plt.figure()
plt.plot(theta_vals*180/pi,basic_airy_beam,  label="basic Airy beam",       c='C0')
plt.axvline(beta_fwhm*180/pi,     label="FWHM",                  c='C1')
plt.axhline(airy_beam_0,        label="max amplitude",         c='C2')
plt.axhline(basic_airy_beam_half_max, label="half of max amplitude", c='C3')
plt.xlabel("theta (deg)")
plt.ylabel("amplitude (arbitrary units)")
plt.title("Basic Airy beam quantities")
plt.legend()
plt.show()

CHORD_ish_fwhm=pi/45. # 4 deg = 4pi/180 rad = pi/45 rad
CHORD_ish_airy_alpha=beta_fwhm/CHORD_ish_fwhm
print('CHORD_ish_airy_alpha=',CHORD_ish_airy_alpha)
CHORD_ish_airy_beam=airybeam(theta_vals,alpha=CHORD_ish_airy_alpha)

plt.figure()
plt.plot(theta_vals*180/pi,CHORD_ish_airy_beam,  label="basic Airy beam",       c='C0')
plt.axvline(beta_fwhm/CHORD_ish_airy_alpha*180/pi,     label="FWHM",                  c='C1')
plt.axhline(airy_beam_0,        label="max amplitude",         c='C2')
plt.axhline(basic_airy_beam_half_max, label="half of max amplitude", c='C3')
plt.xlabel("theta (deg)")
plt.ylabel("amplitude (arbitrary units)")
plt.title("CHORD-ish Airy beam quantities")
plt.legend()
plt.show()