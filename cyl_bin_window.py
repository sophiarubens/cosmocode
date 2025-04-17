import numpy as np
from matplotlib import pyplot as plt
from scipy.special import j1 # first-order Bessel function of the first kind
from scipy.integrate import quad
import time

pi=np.pi
twopi=2.*pi
# THIS USES AN AIRY BEAM WITH GAUSSIAN CHROMATICITY ... WORK THROUGH THE BUGS WITH THIS, BUT GENERALIZE LATER
new_max_n_subdiv=200
new_epsrel=1e-1
ln2=np.log(2)

def thetaHWHM_to_alpha(thetaHWHM):
    npts=2222
    theta_vals=np.linspace(0,twopi,npts)
    basic_airy_beam=(j1(theta_vals)/theta_vals)**2
    basic_airy_beam_half_max=1./8. # derived
    thetaHWHM_ref_airy=theta_vals[np.nanargmin(np.abs(basic_airy_beam-basic_airy_beam_half_max))]
    return thetaHWHM_ref_airy/thetaHWHM

# AIRY BEAM
def theta_airy_arg_real(theta,thetak,thetakp,alpha):
    arg=np.exp(-1j*(thetak-thetakp)*theta)*(j1(alpha*theta)/(alpha*theta))**2*np.sin(theta)
    return arg.real
def theta_airy_arg_imag(theta,thetak,thetakp,alpha):
    arg=np.exp(-1j*(thetak-thetakp)*theta)*(j1(alpha*theta)/(alpha*theta))**2*np.sin(theta)
    return arg.imag
def inner_theta_airy_integral_real(thetak,thetakp,alpha):
    integral,_=quad(theta_airy_arg_real,0,pi,args=(thetak,thetakp,alpha,))
    return integral
def inner_theta_airy_integral_imag(thetak,thetakp,alpha):
    integral,_=quad(theta_airy_arg_imag,0,pi,args=(thetak,thetakp,alpha,))
    return integral
def theta_like_airy(thetak,thetakp,alpha):
    return inner_theta_airy_integral_real(thetak,thetakp,alpha)**2+inner_theta_airy_integral_imag(thetak,thetakp,alpha)**2
# GAUSSIAN BEAM
def theta_gau_arg_real(theta,thetak,thetakp,thetaHWHM):
    arg=np.exp(-1j*(thetak-thetakp)*theta-ln2*(theta/thetaHWHM)**2)
    return arg.real
def theta_gau_arg_imag(theta,thetak,thetakp,thetaHWHM):
    arg=np.exp(-1j*(thetak-thetakp)*theta-ln2*(theta/thetaHWHM)**2)
    return arg.imag
def inner_theta_gau_integral_real(thetak,thetakp,thetaHWHM):
    integral,_=quad(theta_gau_arg_real,0,pi,args=(thetak,thetakp,thetaHWHM,))
    return integral
def inner_theta_gau_integral_imag(thetak,thetakp,thetaHWHM):
    integral,_=quad(theta_gau_arg_imag,0,pi,args=(thetak,thetakp,thetaHWHM,))
    return integral
def theta_like_gau(thetak,thetakp,thetaHWHM):
    return inner_theta_gau_integral_real(thetak,thetakp,thetaHWHM)**2+inner_theta_gau_integral_imag(thetak,thetakp,thetaHWHM)**2


def calc_unnorm_r_like_vec(kparvec,sigLoS,r0): # kpar is a vector of all such modes the instrument sees; can stand in for deltakpar b/c subtraction
    # twopi*sigLoS**2*np.exp(-kparvec**2*sigLoS**2)*(sigLoS**4-2*kparvec**2*sigLoS**6+2*r0*sigLoS**2+kparvec**4*sigLoS**8+r0**4+2*kparvec**2*sigLoS**4*r0**2)
    deltakpar=kparvec-kparvec[0]
    return np.exp(-sigLoS**2*deltakpar**2)

def calc_theta_like_vec(kthetavec,beamtype,thetaHWHM):
    nktheta=len(kthetavec)
    vec=np.zeros(nktheta) # holder for the return b/c quad can't handle vectorized calcs
    for i in range(nktheta):
        if beamtype=="airy":
            alpha=thetaHWHM_to_alpha(thetaHWHM)
            vec[i]=theta_like_airy(kthetavec[i],kthetavec[0],alpha)
        elif beamtype=="gaussian":
            vec[i]=theta_like_gau(kthetavec[i],kthetavec[0],thetaHWHM)
        else:
            assert(1==0), "currently supported beam types are Airy and Gaussian"
    return vec

def W_cyl_binned(kparvec,kthetavec,sigLoS,r0,beamtype,thetaHWHM,save=False):
    unnorm_r_like_vec=calc_unnorm_r_like_vec(kparvec,sigLoS,r0)
    theta_like_vec=calc_theta_like_vec(kthetavec,beamtype,thetaHWHM)
    _,axs=plt.subplots(1,2,figsize=(15,5))
    axs[0].plot(kparvec,unnorm_r_like_vec)
    axs[0].set_xlabel("k$_{||}$")
    axs[0].set_ylabel("amplitude")
    axs[0].set_title("r-like")
    axs[1].plot(kthetavec,theta_like_vec)
    axs[1].set_xlabel("k$_\perp$")
    axs[1].set_ylabel("amplitude")
    axs[1].set_title("theta_like")
    plt.show()
    unnorm_r_like_arr,theta_like_arr=np.meshgrid(unnorm_r_like_vec,theta_like_vec)
    fig,axs=plt.subplots(1,3,figsize=(20,5))
    im=axs[0].imshow(unnorm_r_like_arr,extent=[kparvec[0],kparvec[-1],kparvec[-1],kparvec[0]])
    fig.colorbar(im,ax=axs[0])
    axs[0].set_title("r-like meshgridded array")
    im=axs[1].imshow(theta_like_arr,extent=[kthetavec[0],kthetavec[-1],kthetavec[-1],kthetavec[0]])
    fig.colorbar(im,ax=axs[1])
    axs[1].set_title("theta-like meshgridded array")
    meshed=unnorm_r_like_arr*theta_like_arr # interested in elementwise (not matrix) multiplication
    normed=meshed/np.sum(meshed)
    im=axs[2].imshow(normed,extent=[kthetavec[0],kthetavec[-1],kparvec[-1],kparvec[0]],aspect=kthetavec[-1]/kparvec[-1])
    fig.colorbar(im,ax=axs[2])
    axs[2].set_title("normed product of r-like and theta-like arrs")
    plt.savefig("in_wrapper_cyl_binned_window_check.png")
    plt.show()
    if (save):
        np.save('W_cyl_binned_2D_proxy'+str(time.time())+'.txt',normed)
    return normed