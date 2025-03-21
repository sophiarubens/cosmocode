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

check_inner_r_integral=False
if check_inner_r_integral:
    npts=25
    ubound=104
    sigma=25
    r00=75
    rk_vec=np.linspace(0,ubound,npts)
    scipy_quad_inner_r_integral=np.zeros((npts,npts))
    for i,rk in enumerate(rk_vec):
        for j,rkp in enumerate(rk_vec):
            scipy_quad_inner_r_integral[i,j]=inner_r_integral_real(rk,rkp,sigma,r00)**2+inner_r_integral_imag(rk,rkp,sigma,r00)**2

    rk,rkp=np.meshgrid(rk_vec,rk_vec)
    deltak=rk-rkp
    prefac=2*np.pi*sigma**2*np.exp(-deltak**2*sigma**2)
    long_term=sigma**4-2*deltak**2*sigma**6+2*r00**2*sigma**2+deltak**4*sigma**8+r00**4+2*deltak**2+sigma**4+r00**2
    approximate_inner_r_integral=prefac*long_term

    plt.figure()
    slice_to_inspect=14
    plt.plot(rk_vec,scipy_quad_inner_r_integral[slice_to_inspect],label='scipy')
    plt.plot(rk_vec,approximate_inner_r_integral[slice_to_inspect],linestyle='--',label='approximate')
    plt.xlabel("rk (constant rk')")
    plt.ylabel('mod-squared of inner r integral')
    plt.legend()
    plt.title("rk' slice index "+str(slice_to_inspect))
    plt.savefig('slice_'+str(slice_to_inspect)+'_r_term_r0_'+str(r00)+'.png')
    plt.show()

    fig,axs=plt.subplots(1,3,figsize=(15,5))
    im=axs[0].imshow(scipy_quad_inner_r_integral,extent=[0.,ubound,ubound,0.]) # [L,R,B,T]
    plt.colorbar(im, ax=axs[0])
    axs[0].set_xlabel('$r_k$')
    axs[0].set_ylabel("$r_{k'}$")
    axs[0].set_title('scipy quad')
    im=axs[1].imshow(approximate_inner_r_integral,extent=[0.,ubound,ubound,0.])
    plt.colorbar(im, ax=axs[1])
    axs[1].set_xlabel('$r_k$')
    axs[1].set_ylabel("$r_{k'}$")
    axs[1].set_title('approximate')
    im=axs[2].imshow((approximate_inner_r_integral-scipy_quad_inner_r_integral),extent=[0.,ubound,ubound,0.])
    plt.colorbar(im, ax=axs[2])
    axs[2].set_xlabel('$r_k$')
    axs[2].set_ylabel("$r_{k'}$")
    axs[2].set_title('(approximate-scipy)')
    plt.suptitle('comparison of inner r integrals')
    plt.tight_layout()
    plt.savefig('r_approximate_scipy_quad_comparison.png',dpi=500)
    plt.show()

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

check_inner_phi_integral=False
if check_inner_phi_integral:
    npts=151
    phik_vec=np.linspace(0,2*np.pi,npts)
    scipy_quad_inner_phi_integral=np.zeros((npts,npts))
    for i,phik in enumerate(phik_vec):
        for j,phikp in enumerate(phik_vec):
            scipy_quad_inner_phi_integral[i,j]=inner_phi_integral_real(phik,phikp)**2+inner_phi_integral_imag(phik,phikp)**2

    phik,phikp=np.meshgrid(phik_vec,phik_vec)
    analytical_inner_phi_integral=2.*(1.-np.cos(2*np.pi*(phik-phikp)))/(phik-phikp)**2
    fig,axs=plt.subplots(1,3,figsize=(15,5))
    im=axs[0].imshow(scipy_quad_inner_phi_integral,extent=[0.,2.*np.pi,2.*np.pi,0.]) # [L,R,B,T]
    plt.colorbar(im, ax=axs[0])
    axs[0].set_xlabel('$\phi_k$')
    axs[0].set_ylabel("$\phi_{k'}$")
    axs[0].set_title('scipy quad')
    im=axs[1].imshow(analytical_inner_phi_integral,extent=[0.,2.*np.pi,2.*np.pi,0.])
    plt.colorbar(im, ax=axs[1])
    axs[1].set_xlabel('$\phi_k$')
    axs[1].set_ylabel("$\phi_{k'}$")
    axs[1].set_title('analytical')
    im=axs[2].imshow((analytical_inner_phi_integral-scipy_quad_inner_phi_integral)/scipy_quad_inner_phi_integral,extent=[0.,2.*np.pi,2.*np.pi,0.])
    plt.colorbar(im, ax=axs[2])
    axs[2].set_xlabel('$\phi_k$')
    axs[2].set_ylabel("$\phi_{k'}$")
    axs[2].set_title('(analytical-scipy)/scipy')
    plt.suptitle('comparison of inner phi integrals')
    plt.tight_layout()
    plt.savefig('phi_analytical_scipy_quad_comparison.png',dpi=500)
    plt.show()

def W_binned_airy_beam_entry(rk,rkp,sig,r0,theta_like=theta_like_global,phi_like=phi_like_global): # ONE ENTRY in the kind of W_binned square array that is useful to build
    r_like=inner_r_integral_real(rk,rkp,sig,r0)**2+inner_r_integral_imag(rk,rkp,sig,r0)**2
    deltak=rk-rkp
    longterm=(sig**4-2*deltak**2*sig**6+2*r0*sig**2+deltak**4*sig**8+r0**4+2*deltak**2*sig**4*r0**2)
    # print('longterm=',longterm)
    expterm=twopi*sig**2*np.exp(-deltak**2*sig**2)
    # print('expterm=',expterm)
    r_like_hand=expterm*longterm
    # print('r_like=',r_like)
    # print('r_like_hand',r_like_hand)
    # return 4*pi**2*r_like_hand*theta_like*phi_like
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

npts=25
rkmax=104.
rk_test=np.linspace(0,rkmax,npts)
sig_test=25
r0_test=75
W_binned_airy_beam_array_test=W_binned_airy_beam(rk_test,sig_test,r0_test,verbose=True)

plt.figure()
plt.imshow(W_binned_airy_beam_array_test,extent=[rk_test[0],rk_test[-1],rk_test[-1],rk_test[0]],norm='log',cmap='Reds') # L,R,B,T
plt.colorbar()
plt.xlabel('scalar k; arbitrary units w/ dimensions of 1/L')
plt.ylabel('scalar k-prime; arbitrary units w/ dimensions of 1/L')
plt.title('Spherically binned Airy beam x Gaussian response')
plt.tight_layout()
plt.show()