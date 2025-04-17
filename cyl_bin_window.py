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
    perp_vals=np.linspace(0,twopi,npts)
    basic_airy_beam=(j1(perp_vals)/perp_vals)**2
    basic_airy_beam_half_max=1./8. # derived
    thetaHWHM_ref_airy=perp_vals[np.nanargmin(np.abs(basic_airy_beam-basic_airy_beam_half_max))]
    return thetaHWHM_ref_airy/thetaHWHM

# AIRY BEAM
lbound=-pi/2
ubound= pi/2
def perp_airy_arg_real(thetaperp,deltakperpval,alpha):
    arg=np.exp(-1j*deltakperpval*thetaperp)*(j1(alpha*thetaperp)/(alpha*thetaperp))**2*thetaperp
    return arg.real
def perp_airy_arg_imag(thetaperp,deltakperpval,alpha):
    arg=np.exp(-1j*deltakperpval*thetaperp)*(j1(alpha*thetaperp)/(alpha*thetaperp))**2*thetaperp
    return arg.imag
def inner_perp_airy_integral_real(deltakperpval,alpha):
    integral,_=quad(perp_airy_arg_real,lbound,ubound,args=(deltakperpval,alpha,))
    return integral
def inner_perp_airy_integral_imag(deltakperpval,alpha):
    integral,_=quad(perp_airy_arg_imag,lbound,ubound,args=(deltakperpval,alpha,))
    return integral
def perp_airy(deltakperpval,alpha):
    return inner_perp_airy_integral_real(deltakperpval,alpha)**2+inner_perp_airy_integral_imag(deltakperpval,alpha)**2
# GAUSSIAN BEAM
def perp_gau_arg_real(thetaperp,deltakperpval,thetaHWHM):
    arg=np.exp(-1j*deltakperpval*thetaperp-ln2*(thetaperp/thetaHWHM)**2)*thetaperp
    return arg.real
def perp_gau_arg_imag(thetaperp,deltakperpval,thetaHWHM):
    arg=np.exp(-1j*deltakperpval*thetaperp-ln2*(thetaperp/thetaHWHM)**2)*thetaperp
    return arg.imag
def inner_perp_gau_integral_real(deltakperpval,thetaHWHM):
    integral,_=quad(perp_gau_arg_real,lbound,ubound,args=(deltakperpval,thetaHWHM,))
    return integral
def inner_perp_gau_integral_imag(deltakperpval,thetaHWHM):
    integral,_=quad(perp_gau_arg_imag,lbound,ubound,args=(deltakperpval,thetaHWHM,))
    return integral
def perp_gau(deltakperpval,thetaHWHM):
    return inner_perp_gau_integral_real(deltakperpval,thetaHWHM)**2+inner_perp_gau_integral_imag(deltakperpval,thetaHWHM)**2


def calc_par_vec(deltakparvec,sigLoS,r0):
    prefac=(sigLoS**4-2*deltakparvec**2*sigLoS**6+2*r0*sigLoS**2+deltakparvec**4*sigLoS**8+r0**4+2*deltakparvec**2*sigLoS**4*r0**2)
    expterm=twopi*sigLoS**2*np.exp(-deltakparvec**2*sigLoS**2)
    return prefac*expterm

def calc_perp_vec(deltakperpvec,beamtype,thetaHWHM):
    nkperp=len(deltakperpvec)
    vec=np.zeros(nkperp) # holder for the return b/c quad can't handle vectorized calcs
    if beamtype=="airy":
        perp_fcn=perp_airy
        width_param=thetaHWHM_to_alpha(thetaHWHM)
    elif beamtype=="gaussian":
        perp_fcn=perp_gau
        width_param=thetaHWHM
    else:
        assert(1==0), "currently supported beam types are Airy and Gaussian"
    for i in range(nkperp):
        vec[i]=perp_fcn(deltakperpvec[i],width_param)
    return vec

def W_cyl_binned(deltakparvec,deltakperpvec,sigLoS,r0,beamtype,thetaHWHM,save=False):
    par_vec=calc_par_vec(deltakparvec,sigLoS,r0)
    perp_vec=calc_perp_vec(deltakperpvec,beamtype,thetaHWHM)
    _,axs=plt.subplots(1,2,figsize=(15,5))
    axs[0].plot(deltakparvec,par_vec)
    axs[0].set_xlabel("k$_{||}$")
    axs[0].set_ylabel("amplitude")
    axs[0].set_title("r-like")
    axs[1].plot(deltakperpvec,perp_vec)
    axs[1].set_xlabel("k$_\perp$")
    axs[1].set_ylabel("amplitude")
    axs[1].set_title("perp_like")
    plt.show()
    unnorm_r_arr,perp_arr=np.meshgrid(par_vec,perp_vec)
    fig,axs=plt.subplots(1,3,figsize=(20,5))
    im=axs[0].imshow(unnorm_r_arr,extent=[deltakparvec[0],deltakparvec[-1],deltakparvec[-1],deltakparvec[0]]) # ,norm="log")
    fig.colorbar(im,ax=axs[0])
    axs[0].set_title("r-like meshgridded array")
    im=axs[1].imshow(perp_arr,extent=[deltakperpvec[0],deltakperpvec[-1],deltakperpvec[-1],deltakperpvec[0]]) # ,norm="log")
    fig.colorbar(im,ax=axs[1])
    axs[1].set_title("theta-like meshgridded array")
    meshed=unnorm_r_arr*perp_arr # interested in elementwise (not matrix) multiplication
    normed=meshed/np.sum(meshed)
    im=axs[2].imshow(normed,extent=[deltakperpvec[0],deltakperpvec[-1],deltakparvec[-1],deltakparvec[0]],aspect=deltakperpvec[-1]/deltakparvec[-1]) # ,norm="log")
    fig.colorbar(im,ax=axs[2])
    axs[2].set_title("normed product of r-like and theta-like arrs")
    plt.savefig("in_wrapper_cyl_binned_window_check.png")
    plt.show()
    if (save):
        np.save('W_cyl_binned_2D_proxy'+str(time.time())+'.txt',normed)
    return normed