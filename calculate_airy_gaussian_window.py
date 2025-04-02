import numpy as np
from scipy.special import j1 # first-order Bessel function of the first kind
from scipy.integrate import quad,dblquad
from scipy.linalg import toeplitz
import time

pi=np.pi
twopi=2.*pi
# NUMERICALLY INTEGRATING TO OBTAIN A SPHERICALLY HARMONICALLY BINNED WINDOW FUNCTION
# THIS USES AN AIRY BEAM WITH GAUSSIAN CHROMATICITY ... WORK THROUGH THE BUGS WITH THIS, BUT GENERALIZE LATER
new_max_n_subdiv=200
new_epsrel=1e-1

def theta_arg_real(theta,thetak,thetakp):
    arg=np.exp(-1j*(thetak-thetakp)*theta)*(j1(theta)/theta)**2*np.sin(theta)
    return arg.real
def theta_arg_imag(theta,thetak,thetakp):
    arg=np.exp(-1j*(thetak-thetakp)*theta)*(j1(theta)/theta)**2*np.sin(theta)
    return arg.imag
def inner_theta_integral_real(thetak,thetakp):
    integral,_=quad(theta_arg_real,0,pi,args=(thetak,thetakp,))
    return integral
def inner_theta_integral_imag(thetak,thetakp):
    integral,_=quad(theta_arg_imag,0,pi,args=(thetak,thetakp,))
    return integral
def theta_k_and_kp_arg(thetak,thetakp):
    inner_theta_integral=inner_theta_integral_real(thetak,thetakp)**2+inner_theta_integral_imag(thetak,thetakp)**2
    return inner_theta_integral*np.sin(thetak)*np.sin(thetakp)
theta_like_global,_=dblquad(theta_k_and_kp_arg,0,pi,0,pi)

def phi_arg_real(phi,phik,phikp):
    arg=np.exp(-1j*(phik-phikp)*phi)
    return arg.real
def phi_arg_imag(phi,phik,phikp):
    arg=np.exp(-1j*(phik-phikp)*phi)
    return arg.imag
def inner_phi_integral_real(phik,phikp):
    integral,_=quad(phi_arg_real,0,twopi,args=(phik,phikp,))
    return integral
def inner_phi_integral_imag(phik,phikp):
    integral,_=quad(phi_arg_imag,0,twopi,args=(phik,phikp,))
    return integral
def phi_k_and_kp_arg(phik,phikp):
    return inner_phi_integral_real(phik,phikp)**2+inner_phi_integral_imag(phik,phikp)**2
phi_like_global,_=dblquad(phi_k_and_kp_arg,0,twopi,0,twopi) 

def W_binned_airy_beam_entry_r_hand(rk,rkp,sig,r0,theta_like=theta_like_global,phi_like=phi_like_global):
    deltak=rk-rkp
    r_like_hand=twopi*sig**2*np.exp(-deltak**2*sig**2)
    return 4*pi**2*r_like_hand*theta_like*phi_like

def W_binned_airy_beam_r_hand(rk_vector,sig,r0,save=False,verbose=False): # accumulate the kind of term we're interested in into a square grid
    npts=len(rk_vector)
    row=np.zeros(npts)
    for i in range(npts): # symmetric, circulant
        row[i]=W_binned_airy_beam_entry_r_hand(rk_vector[i],rk_vector[0],sig,r0)
        if verbose:
            print('toeplitz column element {:3}'.format(i))
    arr=toeplitz(row)
    if (save):
        np.save('W_binned_airy_beam_array_'+str(time.time())+'.txt',arr)
    return arr