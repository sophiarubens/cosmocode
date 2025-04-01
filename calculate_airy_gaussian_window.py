import numpy as np
from matplotlib import pyplot as plt
from scipy.special import j1 # first-order Bessel function of the first kind
from scipy.integrate import quad,dblquad
from scipy.linalg import toeplitz
import time
from matplotlib.colors import LogNorm

pi=np.pi
twopi=2.*pi

# NUMERICALLY INTEGRATING TO OBTAIN A SPHERICALLY HARMONICALLY BINNED WINDOW FUNCTION
# THIS USES AN AIRY BEAM WITH GAUSSIAN CHROMATICITY ... WORK THROUGH THE BUGS WITH THIS, BUT GENERALIZE LATER
def r_arg_real(r,rk,rkp,sig,r0):
    arg=np.exp(-1j*(rk-rkp)*r-(((r-r0)**2)/(2*sig**2)))*r**2
    return arg.real
def r_arg_imag(r,rk,rkp,sig,r0):
    arg=np.exp(-1j*(rk-rkp)*r-(((r-r0)**2)/(2*sig**2)))*r**2
    return arg.imag
new_max_n_subdiv=200
new_epsrel=1e-1
def inner_r_integral_real(rk,rkp,sig,r0, tol=1e-1):
    expterm=np.exp(-r0**2/(2.*sig**2))
    # print('real part: np.exp(-r0**2/(2.*sig**2))=',expterm)
    assert(expterm<tol), "expterm="+str(expterm)+"this numerical case is inconsistent with the assumption governing the analytic version to which I am comparing it"
    integral,_=quad(r_arg_real,0,np.infty,args=(rk,rkp,sig,r0,),epsrel=new_epsrel,limit=new_max_n_subdiv)
    return integral
def inner_r_integral_imag(rk,rkp,sig,r0, tol=1e-1):
    expterm=np.exp(-r0**2/(2.*sig**2))
    # print('imag part: np.exp(-r0**2/(2.*sig**2))=',expterm)
    assert(expterm<tol), "expterm="+str(expterm)+"this numerical case is inconsistent with the assumption governing the analytic version to which I am comparing it"
    integral,_=quad(r_arg_imag,0,np.infty,args=(rk,rkp,sig,r0,),epsrel=new_epsrel,limit=new_max_n_subdiv)
    return integral

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

def W_binned_airy_beam_entry(rk,rkp,sig,r0,theta_like=theta_like_global,phi_like=phi_like_global): # ONE ENTRY in the kind of W_binned square array that is useful to build
    r_like=inner_r_integral_real(rk,rkp,sig,r0)**2+inner_r_integral_imag(rk,rkp,sig,r0)**2
    deltak=rk-rkp
    longterm=(sig**4-2*deltak**2*sig**6+2*r0*sig**2+deltak**4*sig**8+r0**4+2*deltak**2*sig**4*r0**2)
    expterm=twopi*sig**2*np.exp(-deltak**2*sig**2)
    r_like_hand=expterm*longterm
    return 4*pi**2*r_like*theta_like*phi_like 

def W_binned_airy_beam(rk_vector,sig,r0,save=True,verbose=False): # accumulate the kind of term we're interested in into a square grid
    npts=len(rk_vector)
    row=np.zeros(npts)
    for i in range(npts): # symmetric, circulant
        row[i]=W_binned_airy_beam_entry(rk_vector[i],rk_vector[0],sig,r0)
        if verbose:
            print('toeplitz column element {:3}'.format(i))
    arr=toeplitz(row)
    if (save):
        np.save('W_binned_airy_beam_array_'+str(time.time())+'.txt',arr)
    return arr