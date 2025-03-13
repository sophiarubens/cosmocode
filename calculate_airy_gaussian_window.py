import numpy as np
from scipy.special import j1 # first-order Bessel function of the first kind
from scipy.integrate import quad,dblquad
import time

pi=np.pi
twopi=2.*pi

# NUMERICALLY INTEGRATING TO OBTAIN A SPHERICALLY HARMONICALLY BINNED WINDOW FUNCTION
# THIS USES AN AIRY BEAM WITH GAUSSIAN CHROMATICITY ... WORK THROUGH THE BUGS WITH THIS, BUT GENERALIZE LATER
r0=25 # another global variable for now ... I don't want to pass r0 because it's not an arg and might screw up quadpack wrapper things?? Although I should test that... the main concern here is slowdown from the flow of eval having to look outside its scope repeatedly
def r_arg_real(r,rk,rkp,sig):
    arg=np.exp(-1j*(rk-rkp)*r-(((r-r0)**2)/(2*sig**2)))*r**2
    return arg.real
def r_arg_imag(r,rk,rkp,sig):
    arg=np.exp(-1j*(rk-rkp)*r-(((r-r0)**2)/(2*sig**2)))*r**2
    return arg.imag
new_max_n_subdiv=200
def inner_r_integral_real(rk,rkp,sig):
    integral,_=quad(r_arg_real,0,np.infty,args=(rk,rkp,sig,),limit=new_max_n_subdiv)
    return integral
def inner_r_integral_imag(rk,rkp,sig):
    integral,_=quad(r_arg_imag,0,np.infty,args=(rk,rkp,sig,),limit=new_max_n_subdiv)
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
print('theta_like_global=',theta_like_global)

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
print('phi_like_global=',phi_like_global)

def W_binned_airy_beam_entry(rk,rkp,sig,theta_like=theta_like_global,phi_like=phi_like_global): # ONE ENTRY in the kind of W_binned square array that is useful to build
    r_like=inner_r_integral_real(rk,rkp,sig)**2+inner_r_integral_imag(rk,rkp,sig)**2
    # print('r_like=',r_like)
    return 4*pi**2*r_like*theta_like*phi_like 

def W_binned_airy_beam(rk_vector,sig,save=True,timeout=600,verbose=False): # accumulate the kind of term we're interested in into a square grid
    earlyexit=False # so far
    t0=time.time()
    npts=len(rk_vector)
    arr=np.zeros((npts,npts))
    element_times=np.zeros(npts**2)
    for i in range(npts):
        for j in range(i,npts):
            t1=time.time()
            arr[i,j]=W_binned_airy_beam_entry(rk_vector[i],rk_vector[j],sig)
            arr[j,i]=arr[i,j] # probably a negligible difference to leave it this way vs. adding an if stmt to manually catch the off-diagonal terms
            t2=time.time()
            element_times[i*npts+j]=t2-t1
            if verbose:
                print('[',i,',',j,'] and [',j,',',i,']')
                # print('populated entries [',i,',',j,'] and [',j,',',i,'] of W_binned_airy_beam in',t2-t1,'s')
            if((t2-t0)>timeout):
                earlyexit=True
                break
        if ((t2-t0)>timeout):
            earlyexit=True
            break
    if (save):
        np.save('W_binned_airy_beam_array_'+str(time.time())+'.txt',arr)
    t3=time.time()
    if verbose:
        if earlyexit:
            print('due to time constraints, upper triangular entries with row-major indices beyond',i-1,',',j-1,' (and their lower triangular symmetric pairs) were not populated')
        print('evaluation took',t3-t0,'s')
        nonzero_element_times=element_times[np.nonzero(element_times)]
        print('eval time per element:',np.mean(nonzero_element_times),'+/-',np.std(nonzero_element_times)) # x2 since half the array doesn't get populated ... easier than using nan-aware quantities
    return arr

npts=20
rkmax=100.
rk_test=np.linspace(0,rkmax,npts)
sig_test=rkmax/2
W_binned_airy_beam_array_test=W_binned_airy_beam(rk_test,sig_test,timeout=20,verbose=True)